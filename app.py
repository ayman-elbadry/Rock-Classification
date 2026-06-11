import os
import torch
import torch.nn as nn
from torchvision import models
import gradio as gr
from PIL import Image
from explicability import get_gradcam

# Classes in alphabetical order as defined in train.py
CLASSES = ['Igneous', 'Metamorphic', 'Sedimentary']

def load_model():
    model_path = 'best_georock_model.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: {model_path} not found. Please train the model first.")
        # We still return the untrained model to let the UI launch, but predictions will be random
        
    model = model.to(device)
    model.eval()
    return model, device

# Load the model globally so it's ready when the app starts
model, device = load_model()

def predict(image):
    if image is None:
        return None, None
    
    if not os.path.exists('best_georock_model.pth'):
        return {"Error": "Model weights not found. Please run train.py first."}, None
        
    # Get Grad-CAM overlay and predictions
    overlay_img, class_idx, probs = get_gradcam(model, image)
    
    # Format probabilities for Gradio Label component
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    
    return prob_dict, overlay_img

# Custom CSS for dark mode and beautiful UI
css = """
body {
    background-color: #121212;
    color: #e0e0e0;
}
.gradio-container {
    max-width: 900px !important;
}
h1 {
    text-align: center;
    color: #4CAF50;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🪨 GeoRock AI - Rock Classification")
    gr.Markdown("Upload an image of a rock to classify it into **Igneous**, **Metamorphic**, or **Sedimentary**.")
    gr.Markdown("The **Grad-CAM** heatmap will show you exactly which textures the AI used to make its decision.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Rock Image")
            submit_btn = gr.Button("Classify Rock", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(label="Prediction Confidence", num_top_classes=3)
            output_cam = gr.Image(type="pil", label="Grad-CAM Heatmap")
            
    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_label, output_cam])
    
    gr.Examples(
        examples=[
            # You can add example image paths here if you have them, e.g.
            # ["data/Dataset/Igneous/Basalt/example.jpg"]
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(share=False)
