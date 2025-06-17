import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights,resnet50,ResNet50_Weights
import torchvision.transforms as transforms
from timm import create_model

# Data avec augmentation de Dataset

train_path="/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
test_path="/kaggle/input/detect-ai-vs-human-generated-images/test.csv"

dataset_path="/kaggle/input/ai-vs-human-generated-dataset"

train_df=pd.read_csv(train_path)
test_df=pd.read_csv(test_path)

image_path=os.path.join(dataset_path,train_df.iloc[0,1])
image=Image.open(image_path)
plt.imshow(image)

train_transforms=transforms.Compose([
    transforms.Resize((232)),                
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),      
    transforms.RandomRotation(10),           
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),        
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

image2=train_transforms(image)
plt.imshow(image2.permute(1,2,0))

test_transforms=transforms.Compose([
    transforms.Resize(232),  
    transforms.CenterCrop(224),             
    transforms.ToTensor(),                        
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
validation_transform=transforms.Compose([
    transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

import gc
import torch

def clear_gpu_memory():
    # Étape 1 : Supprimer toutes les références
    gc.collect()
    
    # Étape 2 : Vider le cache PyTorch
    torch.cuda.empty_cache()
    
    # Étape 3 : Reset CUDA (nettoyage profond)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

# Model

class convnextandswin(nn.Module):
    def __init__(self, num_classes=1):
        super(convnextandswin, self).__init__()

        # Load ConvNeXt Large
        self.convnext = create_model("convnext_large", pretrained=True, num_classes=0)
        convnext_out = self.convnext.num_features

        # Load Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        swin_out = self.swin.num_features

        # Global Average Pooling for each model
        self.global_avg_pooling_convnext = nn.AdaptiveAvgPool1d(1)
        self.global_avg_pooling_swin = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers for feature fusion
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(convnext_out + swin_out),
            nn.Linear(convnext_out + swin_out, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        # Decoder: Additional layers to output classification results
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        # Pass through ConvNeXt and Swin Transformer
        x_convnext = self.convnext(x)
        x_swin = self.swin(x)

        # Debug print statements to inspect tensor shapes
        #print("Shape of x_convnext before pooling:", x_convnext.shape)
        #print("Shape of x_swin before pooling:", x_swin.shape)

        # Apply global average pooling
        x_convnext = self.global_avg_pooling_convnext(x_convnext.unsqueeze(2)).view(x_convnext.size(0), -1)
        x_swin = self.global_avg_pooling_swin(x_swin.unsqueeze(2)).view(x_swin.size(0), -1)

        # Debug print statements to inspect tensor shapes after pooling
        #print("Shape of x_convnext after pooling:", x_convnext.shape)
        #print("Shape of x_swin after pooling:", x_swin.shape)

        # Concatenate both feature vectors
        x_combined = torch.cat((x_convnext, x_swin), dim=1)
        x_fused = self.feature_fusion(x_combined)

        # Pass through the decoder to output the final classification result
        decoded_output = self.decoder(x_fused)

        return decoded_output

# Initialize the model with ConvNeXt Large and Swin Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = convnextandswin(num_classes=1).to(device)

# Freeze All Layers
for param in model.convnext.parameters():
    param.requires_grad = False

for param in model.swin.parameters():
    param.requires_grad = False

# Unfreeze Last 10 Layers
for param in list(model.convnext.parameters())[-20:]:
    param.requires_grad = True

for param in list(model.swin.parameters())[-20:]:
    param.requires_grad=True
model_path="/kaggle/input/fusion-model/best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))



# Data

from torch.utils.data import Dataset, DataLoader
class CustomDataset2(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            # Pas de label en inference
            return img

# Paths to the dataset
from sklearn.model_selection import train_test_split
base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path  = os.path.join(base_dir, 'test.csv')

# Reading the training CSV file
df_train = pd.read_csv(train_csv_path)
# Example of a row: file_name="train_data/041be3153810...", label=0 or 1

# Reading the testing CSV file
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
# Exemple: df_test['id'] = "test_data/e25323c62af644fba97afb846261b05b.jpg", etc.

# Adding the full path to the file_name instead of just "trainORtest_data/xxx.jpg"
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

all_image_paths = df_train['file_name'].values
all_labels = df_train['label'].values

# Splitting train/validation (95% / 5%)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.05,        
    stratify=all_labels,
    random_state=42
)


#train_paths = train_paths[:1500]
#val_paths = val_paths[:150]
print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")
print(train_paths)

train_data = CustomDataset2(train_paths,train_labels, transform=train_transforms)
val_data   = CustomDataset2(val_paths,val_labels ,transform=validation_transform)

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,  num_workers=4,pin_memory=True)
val_loader   = DataLoader(dataset=val_data,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")

for image, label in train_loader:
    print(image.shape)  # Devrait être torch.Size([32, 3, 224, 224])
    print(label.shape)  # Devrait être torch.Size([32])
    break

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, roc_auc_score

# Initialisation
training_param = [p for p in model.parameters() if p.requires_grad]
print(len(training_param))
optimizer = torch.optim.AdamW(training_param, lr=1e-4,weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.BCEWithLogitsLoss()
torch.cuda.empty_cache()
scaler = torch.GradScaler(device='cuda')
print(scaler)

train_losses, train_accuracies = [], []
val_losses, val_accuracies, val_f1_scores, val_roc_aucs = [], [], [], []

epochs = 5
best_val_loss = float('inf')
early_stopping_patience = 7
early_stopping_counter = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    model.train()
    
    for data, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
        data = data.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        epoch_accuracy += (preds == labels).float().mean().item()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss/len(train_loader))
    train_accuracies.append(epoch_accuracy/len(train_loader))

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_accuracy = 0.0
    val_pred_class = []
    val_label_class = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc=f"validation_epoch {epoch+1}"):
            data, labels = data.to(device), labels.to(device).float()
            
            with torch.autocast(device_type='cuda'):
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, labels)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_val_accuracy += (preds == labels).float().mean().item()
            epoch_val_loss += loss.item()
            val_pred_class.extend(preds.cpu().numpy())
            val_label_class.extend(labels.cpu().numpy())
    
    val_loss = epoch_val_loss/len(val_loader)
    val_acc = epoch_val_accuracy/len(val_loader)
    val_f1 = f1_score(np.array(val_label_class), np.array(val_pred_class))
    val_roc_auc = roc_auc_score(np.array(val_label_class), np.array(val_pred_class))

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)
    val_roc_aucs.append(val_roc_auc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
          f"Train Acc: {epoch_accuracy/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Val ROC AUC: {val_roc_auc:.4f}")
    
    scheduler.step()

    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Plot Loss and Accuracy

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot Loss
axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
axs[0].set_title('Loss per Epoch')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot Accuracy
axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
axs[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='red')
axs[1].set_title('Accuracy per Epoch')

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# Show the plots
plt.tight_layout()
plt.savefig('Accuracy per Epoch.png')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Compute ROC curve
fpr, tpr, _ = roc_curve(np.array(val_label_class , dtype=int), np.array(val_pred_class, dtype=int))

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {val_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# Generate and plot Confusion Matrix
conf_matrix = confusion_matrix(val_label_class , val_pred_class)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print Classification Report
print("\nClassification Report:")
print(classification_report(val_label_class , val_pred_class))


base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
test_csv_path  = os.path.join(base_dir, 'test.csv')
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
class TestImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

test_dataset = TestImageDataset(df_test['id'].values, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
        
        # On récupère la classe prédite
        output = model(data).squeeze(1)  # Ensure correct dimensions

        # Apply sigmoid activation to convert logits to probabilities
        probs = torch.sigmoid(output)

        # Convert probabilities to binary class (0 or 1) using threshold 0.5
        preds = (probs > 0.5).int()
        
        predictions.extend(preds.cpu().numpy())
        image_names.extend([f"test_data_v2/{name}" for name in names])

# Créer le DataFrame au format "id,label"
submission_df = pd.DataFrame({
    'id': image_names,
    'label': predictions
})

submission_df.head()
submission_df.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")