# Neural Network Visualizer (CIFAR-10 with ResNet)

## Overview

This project provides an interactive visualization and analysis of a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset using PyTorch and ResNet-18. The notebook demonstrates how to train, evaluate, and visualize the internal workings of a deep neural network, including feature activations and gradient-based explanations (Grad-CAM).

## Features
- **Data Pipeline:** Loads and preprocesses CIFAR-10 images, resizing and normalizing them for ResNet input.
- **Model:** Utilizes a pre-trained ResNet-18, fine-tuned for 10-class classification.
- **Training:** Includes a training loop with loss tracking and model saving.
- **Layer Visualization:**
  - Registers hooks to capture activations and gradients from various layers.
  - Visualizes feature maps (activation maps) from shallow to deep layers, showing how the network processes images.
- **Logit Analysis:**
  - Plots the output of the fully connected layer (class logits) as bar charts, highlighting the predicted class and confidence.
- **Grad-CAM:**
  - Implements Grad-CAM to generate heatmaps that highlight image regions most influential for the model's predictions.

## Setup

1. **Install Dependencies**
   - The notebook requires Python 3, PyTorch, torchvision, matplotlib, seaborn, numpy, and OpenCV.
   - Install dependencies with:
     ```bash
     pip install torch torchvision matplotlib seaborn numpy opencv-python
     ```

2. **Download CIFAR-10 Dataset**
   - The dataset is automatically downloaded when running the notebook.

## Usage

1. **Open the Notebook**
   - Launch Jupyter Notebook or JupyterLab and open `nn-visualizer.ipynb`.

2. **Run All Cells**
   - Execute the cells sequentially to:
     - Set up the data pipeline
     - Load and modify the ResNet-18 model
     - Train the model (or load a pre-trained checkpoint)
     - Register hooks for activations and gradients
     - Visualize feature maps and Grad-CAM heatmaps

3. **Visualizations**
   - **Activation Maps:** See how different layers respond to input images, from basic edges to complex object parts.
   - **Logit Bar Charts:** Inspect the model's confidence and class predictions.
   - **Grad-CAM:** Understand which image regions drive the model's decisions.

## Example Visualizations
- **Activation Maps:**
  - Shallow layers capture edges and textures.
  - Deeper layers focus on object parts and high-level features.
- **Grad-CAM:**
  - Heatmaps overlayed on input images show the most influential regions for predictions.

## File Structure
- `nn-visualizer.ipynb` — Main notebook with code, training, and visualizations.
- `resnet_cifar10.pth` — Saved model weights after training.

## Credits
- Built with PyTorch and torchvision.
- Grad-CAM implementation inspired by research on neural network interpretability.

## License
This project is for educational and research purposes. Please cite appropriately if used in academic work.