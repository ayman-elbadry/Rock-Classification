# GeoRock AI - Rock Classification

A complete Deep Learning pipeline using PyTorch and ResNet50 to classify rocks into 3 main categories: Igneous, Metamorphic, and Sedimentary. The project includes Grad-CAM explainability and a Web UI using Gradio.

## Features
- **Transfer Learning**: Uses a pre-trained ResNet50 model.
- **Handling Imbalanced Data**: Implements `WeightedRandomSampler` for balanced training.
- **Explainable AI**: Integrates Grad-CAM to visualize which parts of the rock texture the model focuses on.
- **Web Interface**: Easy-to-use Gradio app for real-time inference and visualization.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset is located at `data/Dataset/` with the following structure:
   ```
   data/Dataset/
   ├── Igneous/
   │   ├── Basalt/
   │   └── Granite/
   ├── Metamorphic/
   │   ├── Marble/
   │   └── Quartzite/
   └── Sedimentary/
       ├── Coal/
       ├── Limestone/
       └── Sandstone/
   ```

## Usage

### Training the Model
Run the training script to fine-tune the ResNet50 model on your dataset:
```bash
python train.py
```
This will generate a `best_georock_model.pth` file containing the best model weights.

### Launching the Web Interface
Start the Gradio application:
```bash
python app.py
```
Open your browser to the local URL provided by Gradio to use the app.
