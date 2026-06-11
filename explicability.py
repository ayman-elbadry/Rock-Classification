import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        self.target_layer.register_backward_hook(self.save_gradient)
        # Hook for activations
        self.target_layer.register_forward_hook(self.save_activation)
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[:, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Calculate weights (Global Average Pooling on gradients)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only positive influences
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, output.cpu().data.numpy()[0], class_idx

def overlay_cam_on_image(img, cam_mask, alpha=0.5):
    """
    Overlay Grad-CAM mask on the original image.
    img: PIL Image or numpy array (RGB)
    cam_mask: numpy array (H, W) between 0 and 1
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    # Resize cam_mask to match image size
    cam_mask = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
    
    # Apply colormap (JET)
    heatmap = cm.jet(cam_mask)[..., :3] # RGBA to RGB
    
    # Convert heatmap to uint8
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Superimpose heatmap onto image
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

def get_gradcam(model, image, target_layer=None):
    """
    Helper function to process an image and return Grad-CAM results.
    """
    if target_layer is None:
        # Default target layer for ResNet50
        target_layer = model.layer4[-1]
        
    grad_cam = GradCAM(model, target_layer)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        
    input_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    cam_mask, logits, class_idx = grad_cam(input_tensor)
    
    probs = F.softmax(torch.tensor(logits), dim=0).numpy()
    
    overlay_img = overlay_cam_on_image(image, cam_mask)
    
    return overlay_img, class_idx, probs
