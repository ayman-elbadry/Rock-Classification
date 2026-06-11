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

def get_dataloaders(data_dir, batch_size=32):
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, full_dataset.class_to_idx

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
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
    return model

def main():
    data_dir = 'data/Dataset'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Preparing data loaders...")
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir, batch_size=32)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print("Classes found:", class_to_idx)
    
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
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5, patience=5, device=device)
    
    print("Unfreezing all layers for fine-tuning...")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    # Optimizer for all layers with a lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting fine-tuning...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20, patience=5, device=device)
    
    print("Training complete! Best model saved as 'best_georock_model.pth'.")

if __name__ == '__main__':
    main()
