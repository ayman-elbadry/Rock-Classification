import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

class GeoRockDataset(Dataset):
    def __init__(self, root_dir, transform=None, file_paths=None, labels=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if file_paths is None or labels is None or class_to_idx is None:
            self.classes = sorted(['Igneous', 'Metamorphic', 'Sedimentary'])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            self.file_paths = []
            self.labels = []
            
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                # Traverse subdirectories (e.g., Basalt, Granite inside Igneous)
                for subtype in os.listdir(class_dir):
                    subtype_dir = os.path.join(class_dir, subtype)
                    if not os.path.isdir(subtype_dir):
                        continue
                        
                    for img_name in os.listdir(subtype_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.file_paths.append(os.path.join(subtype_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
        else:
            self.file_paths = file_paths
            self.labels = labels
            self.class_to_idx = class_to_idx
            self.classes = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=8):
    # Full dataset purely for extracting paths and labels
    full_dataset = GeoRockDataset(root_dir=data_dir)
    
    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        full_dataset.file_paths, full_dataset.labels, test_size=0.2, stratify=full_dataset.labels, random_state=42
    )
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = GeoRockDataset(root_dir=data_dir, transform=train_transform, file_paths=train_paths, labels=train_labels, class_to_idx=full_dataset.class_to_idx)
    val_dataset = GeoRockDataset(root_dir=data_dir, transform=val_transform, file_paths=val_paths, labels=val_labels, class_to_idx=full_dataset.class_to_idx)
    
    # Handle Class Imbalance with WeightedRandomSampler
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, full_dataset.class_to_idx

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Record history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f'New best model found! Saving weights.')
                    torch.save(best_model_wts, 'best_georock_model.pth')
                else:
                    epochs_no_improve += 1
                    
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
            
        print()
        
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_curves(history, output_dir='training_results'):
    """Plot les courbes de loss et accuracy pour train/val."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=4)
    ax1.set_title('Loss par Époque', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy', markersize=4)
    ax2.set_title('Accuracy par Époque', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Courbes d'entraînement sauvegardées (training_curves.png)")


def plot_confusion_matrix(model, dataloader, class_names, device, output_dir='training_results'):
    """Génère et sauvegarde la matrice de confusion sur le jeu de validation."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Matrice de confusion (valeurs absolues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax1, cbar=True)
    ax1.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Prédictions')
    ax1.set_ylabel('Vrais Labels')
    
    # Matrice de confusion normalisée (%)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Oranges', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, cbar=True, vmin=0, vmax=100)
    ax2.set_title('Matrice de Confusion Normalisée (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prédictions')
    ax2.set_ylabel('Vrais Labels')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("-> Matrice de confusion sauvegardée (confusion_matrix.png)")
    
    return all_labels, all_preds


def plot_classification_report(all_labels, all_preds, class_names, output_dir='training_results'):
    """Génère le rapport de classification et sauvegarde un résumé visuel."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Print text report to console
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(report_text)
    
    # Save text report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Generate visual report
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for i, metric in enumerate(metrics):
        values = [report_dict[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1-Score par Classe', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add overall accuracy as text
    overall_acc = report_dict['accuracy']
    ax.text(0.98, 0.02, f'Accuracy globale: {overall_acc:.2%}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("-> Rapport de classification sauvegardé (classification_report.png + .txt)")


def plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir='training_results'):
    """Diagramme circulaire montrant la distribution des prédictions correctes par classe."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    correct_per_class = []
    total_per_class = []
    for i in range(len(class_names)):
        mask = all_labels == i
        total_per_class.append(mask.sum())
        correct_per_class.append((all_preds[mask] == i).sum())
    
    accuracies = [c / t * 100 if t > 0 else 0 for c, t in zip(correct_per_class, total_per_class)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of per-class accuracy
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = ax1.bar(class_names, accuracies, color=colors, edgecolor='gray')
    ax1.set_title('Accuracy par Classe (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Pie chart of correct predictions distribution
    ax2.pie(correct_per_class, labels=class_names, autopct='%1.1f%%', colors=colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    ax2.set_title('Distribution des Prédictions Correctes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("-> Accuracy par classe sauvegardée (per_class_accuracy.png)")

def main():
    data_dir = 'data/Dataset'
    output_dir = 'training_results'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Preparing data loaders...")
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir, batch_size=8)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print("Classes found:", class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Setup ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze all layers except the final fully connected layer initially
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) # 3 classes
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimize only the classifier layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    print("Starting training...")
    # Train the classifier for a few epochs
    model, history_phase1 = train_model(model, dataloaders, criterion, optimizer, num_epochs=5, patience=5, device=device)
    
    print("Unfreezing all layers for fine-tuning...")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    # Optimizer for all layers with a lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting fine-tuning...")
    model, history_phase2 = train_model(model, dataloaders, criterion, optimizer, num_epochs=20, patience=5, device=device)
    
    print("Training complete! Best model saved as 'best_georock_model.pth'.")
    
    # Merge training history from both phases
    full_history = {
        key: history_phase1[key] + history_phase2[key]
        for key in history_phase1
    }
    
    # ========== Post-Training Visualizations ==========
    print("\n" + "="*60)
    print("GÉNÉRATION DES GRAPHIQUES POST-ENTRAÎNEMENT")
    print("="*60)
    
    # 1. Training curves (loss + accuracy)
    plot_training_curves(full_history, output_dir)
    
    # 2. Confusion matrix
    all_labels, all_preds = plot_confusion_matrix(model, val_loader, class_names, device, output_dir)
    
    # 3. Classification report (precision, recall, f1)
    plot_classification_report(all_labels, all_preds, class_names, output_dir)
    
    # 4. Per-class accuracy
    plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir)
    
    print(f"\nTous les graphiques sont sauvegardés dans le dossier : {output_dir}/")

if __name__ == '__main__':
    main()

