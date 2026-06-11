import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def perform_eda(data_dir="data/Dataset", output_dir="eda_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    classes = sorted(['Igneous', 'Metamorphic', 'Sedimentary'])
    
    class_counts = {}
    subclass_counts = {}
    sample_images = {}
    
    print("Analyse du dataset en cours...")
    
    # Collect data
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_counts[class_name] = 0
        subclass_counts[class_name] = {}
        
        for subclass in os.listdir(class_dir):
            subclass_dir = os.path.join(class_dir, subclass)
            if not os.path.isdir(subclass_dir):
                continue
                
            images = [img for img in os.listdir(subclass_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            
            subclass_counts[class_name][subclass] = count
            class_counts[class_name] += count
            
            # Save one sample image per class if not already saved
            if class_name not in sample_images and count > 0:
                sample_images[class_name] = os.path.join(subclass_dir, images[0])
                
    # 1. Plot Class Distribution
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Distribution des Classes Principales de Roches')
    plt.xlabel('Classes')
    plt.ylabel("Nombre d'images")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')
        
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    print("-> Graphique des classes sauvegardé (class_distribution.png)")
    
    # 2. Plot Subclass Distribution
    plt.figure(figsize=(12, 6))
    all_subclasses = []
    all_counts = []
    colors = []
    color_map = {'Igneous': '#ff9999', 'Metamorphic': '#66b3ff', 'Sedimentary': '#99ff99'}
    
    for cls in classes:
        for sub, count in subclass_counts[cls].items():
            all_subclasses.append(f"{sub}\n({cls})")
            all_counts.append(count)
            colors.append(color_map[cls])
            
    bars2 = plt.bar(all_subclasses, all_counts, color=colors)
    plt.title('Distribution des Sous-Types de Roches')
    plt.xlabel('Sous-types')
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subclass_distribution.png'))
    plt.close()
    print("-> Graphique des sous-classes sauvegardé (subclass_distribution.png)")
    
    # 3. Plot Sample Images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Exemples d\'images par classe', fontsize=16)
    
    for ax, (cls, img_path) in zip(axes, sample_images.items()):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()
    print("-> Grille d'exemples sauvegardée (sample_images.png)")
    
    print(f"EDA terminée avec succès ! Tous les graphiques sont dans le dossier : {output_dir}/")

if __name__ == "__main__":
    perform_eda()
