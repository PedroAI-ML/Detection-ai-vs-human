
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

def load_data():
    train_path="/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
    test_path="/kaggle/input/detect-ai-vs-human-generated-images/test.csv"

    dataset_path="/kaggle/input/ai-vs-human-generated-dataset"

    # on cherche à lire les fichier csv, donc on cherche à appliquer pd.read_csv

    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)

    #img=train_df.iloc[0,1]
    #image_path=os.path.join(dataset_path, img)


    #pic=Image.open(image_path)
    #plt.imshow(pic)
    #plt.axis("off")
    #plt.show()
    return train_df, test_df,dataset_path




train_df, test_df, dataset_path=load_data()
print(test_df.shape)
img=test_df.iloc[1,0]
img=os.path.join(dataset_path,img)

pic=Image.open(img)
plt.imshow(pic)
plt.axis("off")
plt.show()

display(train_df.head())
print("--------------------------")
print(test_df.head())
print("---------------------------")



submission_path="/kaggle/input/solution/submission.csv"
solution_df=pd.read_csv(submission_path)

img=solution_df.iloc[1,0]
img=os.path.join(dataset_path,img)

pic=Image.open(img)
plt.imshow(pic)
plt.axis("off")
plt.show()
display(solution_df.head)

définir le filtre passe bande

model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.resquires_grad=False
num=model.fc.in_features
model.fc=nn.Sequential(
    nn.Linear(num,512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512,1)
)
for param in model.layer4.parameters():
    param.required_grad=True
for param in model.fc.parameters():
    param.requires_grad=True
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



from torchvision.transforms import InterpolationMode
import albumentations.core.composition
import albumentations.augmentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_transform=transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    #transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

 

train_transform2 = transforms.Compose([  
    transforms.Resize(256),  # Augmentation légère avant crop  
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33), antialias=True),  
    transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.1),  
    transforms.RandomGrayscale(p=0.2),  
    transforms.RandomErasing(scale=(96/224, 96/224), ratio=(1, 1), value=128, p=0.2),  
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Ajusté pour rester raisonnable  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])   


#strong_transform = albumentations.core.composition.Compose([
#    A.geometric.resize.SmallestMaxSize(max_size=512),
#    A.geometric.transforms.HorizontalFlip(p=0.5),
#    A.crops.transforms.RandomResizedCrop(height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.0/0.75), p=0.2),
#    A.crops.RandomCrop(height=512, width=512),
#    A.transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.1, p=0.8),
#    A.transforms.ToGray(p=0.2),
#    A.dropout.CoarseDropout(max_holes=1, min_holes=1, hole_height_range=(96, 96), hole_width_range=(96, 96), fill_value=128, p=0.2),
#    A.transforms.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#    A.GaussianBlur(),
    
#    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#    ToTensorV2()
#])



validation_transform=transforms.Compose([
    transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

test_transform=transforms.Compose([
    transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])
#print(img)
#image=cv2.imread(img)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#augmented = strong_transform(image=image)
#transformed_image = augmented["image"]
#transformed_image = transformed_image.permute(1, 2, 0).numpy()

# Afficher l'image originale et transformée
#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#axes[0].imshow(image)
#axes[0].set_title("Image originale")
#axes[1].imshow(transformed_image)
#axes[1].set_title("Image transformée")
#plt.show()

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Charger le CSV
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]  # Supposons que la première colonne est le chemin de l'image
        dataset_path="/kaggle/input/ai-vs-human-generated-dataset"

        img_path=os.path.join(dataset_path,img_path)
        label = self.data.iloc[idx, 2]    # Supposons que la deuxième colonne est le label (0 ou 1)

        # Charger l'image
        image = Image.open(img_path).convert("RGB")

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label
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

class CustomDataset3(Dataset):
    def __init__(self, file_list, labels=None, transform=None,strong_transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.strong_transform=strong_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(img)
        if strong_transform:
            augmented=self.strong_transform(image=image)
            image=augmented["image"]
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            # Pas de label en inference
            return image

# Charge le dataset
train_dataset = CustomDataset(csv_file="/kaggle/input/detect-ai-vs-human-generated-images/train.csv", transform=train_transform)
# Créer un DataLoader pour l'entraînement
print(len(train_dataset))
# Vérifier si ça fonctionne


#from torch.utils.data import random_split
#from torch.utils.data import random_split, DataLoader

# Définir la taille des ensembles
#train_size = int(0.9 * len(train_dataset))  # 90% pour l'entraînement
#val_size = len(train_dataset) - train_size  # 10% pour la validation

# Découper train_dataset en train et validation
#train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#val_loader=DataLoader(val_dataset,batch_size=64,shuffle=False)

#for image, label in train_loader:
#    print(image.shape)  # Devrait être torch.Size([32, 3, 224, 224])
#    print(label.shape)  # Devrait être torch.Size([32])
#    break

#print(len(train_dataset))
#print(len(val_dataset))


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

train_data = CustomDataset2(train_paths,train_labels, transform=train_transform)
val_data   = CustomDataset2(val_paths,val_labels ,transform=validation_transform)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True,  num_workers=4)
val_loader   = DataLoader(dataset=val_data,   batch_size=64, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")

for image, label in train_loader:
    print(image.shape)  # Devrait être torch.Size([32, 3, 224, 224])
    print(label.shape)  # Devrait être torch.Size([32])
    break

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


train_losses, train_accuracies = [], []
val_losses, val_accuracies, val_f1_scores, val_roc_aucs = [], [], [], []

epochs = 2
best_val_loss = float('inf')
early_stopping_patience = 5
early_stopping_counter = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    model.train()
    
    for data, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
        data = data.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
       
        outputs = model(data).squeeze(1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
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

torch.save(model.state_dict(), "model_weights.pth")

#torch.save(model, "model.pth")

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

test_dataset = TestImageDataset(df_test['id'].values, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)



model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
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

# Sauvegarder les poids
torch.save(model.state_dict(), 'model_weights.pth')

# Déplacer le fichier vers le dossier d'output
import shutil
shutil.move('model_weights.pth', '/kaggle/working/model_weights.pth')
