# Rapport de Projet : GeoRock AI

**Objectif** : Développer un modèle de Deep Learning capable de classifier des images de roches en 3 catégories (Magmatiques, Sédimentaires, Métamorphiques), avec un module d'explicabilité Grad-CAM, le tout packagé avec une interface utilisateur et versionné sur GitHub.

Ce document retrace les grandes étapes du projet, les obstacles rencontrés et les solutions techniques qui y ont été apportées.

---

## Étape 1 : Analyse Exploratoire des Données (EDA)

L'exploration initiale du dataset a mis en évidence la structure suivante :
- Les images n'étaient pas rangées directement dans les 3 classes principales, mais dans des **sous-dossiers de sous-types** (ex: `Igneous/Basalt/`, `Sedimentary/Limestone/`).
- Le volume total d'images :
  - **Igneous** : 187 images
  - **Metamorphic** : 864 images
  - **Sedimentary** : 1032 images

> [!WARNING]
> **Problème rencontré : Déséquilibre massif des classes (Class Imbalance)**
> Le ratio entre la classe la plus faible (Igneous) et la plus forte (Sedimentary) était de 1:5.5. Un entraînement naïf aurait poussé le modèle à prédire "Sedimentary" par défaut pour minimiser mathématiquement sa perte.

> [!TIP]
> **Solution appliquée :**
> Nous avons généré un script d'EDA `eda.py` pour visualiser cette répartition.
> Côté entraînement, nous avons implémenté un `WeightedRandomSampler` de PyTorch. Il calcule le poids inverse de chaque classe ($W_c = 1 / N_c$) et force le DataLoader à sur-échantillonner les images "Igneous" et sous-échantillonner les "Sedimentary" à chaque epoch, assurant ainsi un apprentissage équilibré.

*(Voir les graphiques générés dans le dossier `eda_results/`)*

---

## Étape 2 : Création du Pipeline d'Entraînement (`train.py`)

Nous avons utilisé **Transfer Learning** avec une architecture **ResNet50** pré-entraînée sur ImageNet, ce qui permet d'exploiter des filtres d'extraction de features robustes sans avoir besoin de dizaines de milliers d'images.

> [!WARNING]
> **Problème rencontré : Chargement du dataset atypique**
> Le module standard `torchvision.datasets.ImageFolder` s'attend à `racine/classe/image.jpg`. Notre dataset était `racine/classe/sous-classe/image.jpg`, ce qui faussait les labels.

> [!TIP]
> **Solution appliquée :**
> Création de la classe custom `GeoRockDataset`. Ce Custom Dataset parcourt l'arbre des fichiers de façon dynamique pour extraire uniquement les classes parentes (le niveau N+1 par rapport à la racine) tout en ignorant le niveau intermédiaire des sous-classes lors du mapping des labels (0, 1, 2).

**Stratégie de Fine-Tuning** :
1. "Gel" (Freeze) de tout le réseau ResNet50 sauf la dernière couche linéaire (`fc`).
2. Entraînement pendant 5 epochs pour spécialiser la dernière couche.
3. "Dégel" (Unfreeze) de l'ensemble du réseau avec un "Learning Rate" divisé par 10 (`1e-4`) pour peaufiner l'intégralité des couches convolutions sans détruire les poids pré-entraînés.

---

## Étape 3 : Module d'Explicabilité (`explicability.py`)

Les réseaux de neurones profonds sont souvent critiqués pour être des "boîtes noires". Pour apporter de la confiance à la prédiction, il fallait implémenter **Grad-CAM**.

> [!WARNING]
> **Problème rencontré : Extraction des gradients sur ResNet50**
> Pour calculer la Grad-CAM, on doit récupérer les activations de la dernière couche de convolution et les gradients du backward pass, qui ne sont normalement pas gardés en mémoire par PyTorch pour économiser la RAM.

> [!TIP]
> **Solution appliquée :**
> Utilisation des "Hooks" PyTorch (`register_forward_hook` et `register_backward_hook`). Le script s'accroche spécifiquement à `model.layer4[-1]` (dernière convolution du ResNet50) pour intercepter le signal. Nous générons ainsi une *heatmap* colorisée (palette Jet) superposée à l'image d'origine.

---

## Étape 4 : Interface Utilisateur (`app.py`)

Pour rendre le modèle accessible :
- **Outil choisi** : `Gradio` (plus direct que Streamlit pour des pipelines purement orientés Computer Vision/Inputs d'images).
- Fonctionnalités : L'utilisateur téléverse une image, et le script retourne instantanément la probabilité de prédiction des 3 classes ainsi que l'image superposée générée par `explicability.py`.

---

## Étape 5 : Gestion de Version et MLOps (Git)

> [!WARNING]
> **Problème rencontré : Fichiers lourds et Git**
> Le dataset "data/" pèse lourd et le fichier des poids `.pth` peut dépasser les 100Mo, ce qui causerait l'échec d'un `git push` standard (limite GitHub à 100Mo par fichier) ou ralentirait inutilement le dépôt.

> [!TIP]
> **Solution appliquée :**
> Génération d'un fichier `.gitignore` robuste excluant explicitement `data/` et `*.pth`.
> Les commandes git (`init`, `add`, `commit`, `remote add`, `push`) ont été scriptées et exécutées via le terminal pour envoyer proprement la logique du code (et non les données) sur la branche principale (`main`) de votre GitHub.

---

## Conclusion
Le projet dispose aujourd'hui d'un socle d'apprentissage solide (gestion du déséquilibre), d'un mécanisme de compréhension transparent (Grad-CAM), et d'une architecture code propre et versionnée. 
Prochaines évolutions possibles : Tester des modèles plus légers (ex: MobileNet) si le temps d'inférence web devenait un problème.
