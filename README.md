# Detect AI vs Human-Generated Images

## Présentation

Ce projet vise à distinguer automatiquement les images générées par intelligence artificielle (IA) des images réelles, en utilisant des techniques avancées de deep learning, de fusion de modèles et d’analyse fréquentielle[1][2].

---

## Objectifs

- Classer automatiquement une image comme “réelle” ou “générée par IA”[1].
- Analyser les différences entre images IA et humaines (textures, fréquences, artefacts)[2].
- Comparer plusieurs architectures modernes : ResNet50, ConvNeXt + Swin Transformer, VAE-FIRE[1][2].

---

## Dataset

- **Source** : Compétition Kaggle “AI vs Human Generated Images”[1].
- **Entraînement** : 79 950 images (50% IA, 50% réelles)[1].
- **Test** : 19 986 images[1].
- **Origine IA** : Stable Diffusion, MidJourney, DALL-E[2].
- **Origine réelle** : Pexels, Unsplash, WikiArt[2].

---

## Approches et Modèles

### 1. ResNet50 (Baseline)
- Backbone pré-entraîné sur ImageNet, fine-tuning sur la tête de classification[1].
- Optimiseur : AdamW, Scheduler : Cosine Annealing, Loss : BCEWithLogitsLoss[2].
- Augmentations : flips, crops, jitter couleur, blur, dropout[2].
- **Résultats** : Accuracy 91-93%, F1-score 0.92, AUC 0.95[1].

### 2. Fusion ConvNeXt + Swin Transformer
- ConvNeXt Large pour les détails locaux, Swin Transformer Base pour le contexte global[2].
- Fusion des features, classification via plusieurs couches fully connected[2].
- Mixed Precision pour accélérer l’entraînement[2].
- **Résultats** : Accuracy 95%, F1-score 0.95, AUC 0.95[2].

### 3. Méthode FIRE (Frequency-guided Reconstruction Error) avec VAE
- Analyse fréquentielle : masquage FFT, reconstruction par VAE, classification sur l’erreur[2].
- **Résultats** : AUC ~0.69 (dépend de la taille du dataset et de la puissance GPU)[2].

---

## Structure du Dépôt

detect-ai-vs-human/
│
├── README.md
├── requirements.txt
├── data/
│ ├── train.csv
│ └── test.csv
├── src/
│ ├── train_resnet.py
│ ├── train_fusion.py
│ ├── train_fire.py
│ ├── dataset.py
│ └── utils.py
├── notebooks/
│ └── exploration.ipynb
├── models/
│ ├── best_resnet.pth
│ ├── best_fusion.pth
│ └── best_fire.pth
└── submission/
└── submission.csv


---

## Installation & Prérequis

1. **Cloner le dépôt :**
git clone https://github.com/mon-utilisateur/detect-ai-vs-human.git
cd detect-ai-vs-human

text
2. **Installer les dépendances :**
pip install -r requirements.txt

text
Principaux packages : torch, torchvision, albumentations, pandas, numpy, scikit-learn, diffusers, tqdm, pillow, matplotlib[1][2].

3. **Télécharger les datasets** (voir instructions dans `data/` ou sur Kaggle)[1].

---

## Utilisation

### Entraînement d’un modèle ResNet50

python src/train_resnet.py --epochs 30 --batch_size 64



### Entraînement du modèle de fusion ConvNeXt+Swin

python src/train_fusion.py --epochs 30 --batch_size 32



### Entraînement du modèle FIRE (VAE + FFT)

python src/train_fire.py --epochs 10 --batch_size 4



### Génération d’une soumission Kaggle

python src/generate_submission.py --model_path models/best_resnet.pth



---

## Résultats

| Modèle                | Accuracy | F1-score | AUC   | Points forts                  |
|-----------------------|----------|----------|-------|-------------------------------|
| ResNet50              | 91-93%   | 0.92     | 0.95  | Robuste, rapide               |
| ConvNeXt + Swin       | 95%      | 0.95     | 0.95  | Équilibre local/global         |
| FIRE (VAE + FFT)      | ~69%     | -        | 0.69  | Approche originale, perfectible|

---

## Conseils de Reproductibilité

- GPU recommandé (16 Go minimum pour la méthode FIRE)[2].
- Vérifier la cohérence des labels (attention aux biais de résolution ou de compression)[2].
- Nettoyer le dataset pour supprimer les biais techniques[2].
- Tester les modèles sur des images “borderline” pour améliorer la robustesse[2].

---

## Pour aller plus loin

- Tester des architectures hybrides (ex : CLIP + Swin + ConvNeXt)[2].
- Explorer des techniques d’augmentation avancées (CutMix, MixUp)[2].
- Améliorer la gestion mémoire pour FIRE et entraîner sur des datasets plus larges[2].

---

## Références

- Kaggle Competition: AI vs Human Generated Images[1].
- Rapport de projet ENSEIRB-Matmeca 2025[2].
- Notebooks et scripts détaillés dans le dossier `notebooks/`[2].

---

**N’hésitez pas à ouvrir une issue ou une pull request pour toute suggestion ou amélioration !**
