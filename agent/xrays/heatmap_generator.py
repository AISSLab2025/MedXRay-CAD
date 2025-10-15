import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from torchvision import models, transforms
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define CLAHE Transform class
class CLAHETransform:
    def __init__(self, clip_limit=0.10, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Convert PIL image to numpy array if necessary
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # If the image is RGB (3 channels), convert it to LAB color space
        if img.ndim == 3:
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_img)

            # Apply CLAHE only to the L (lightness) channel
            l_channel = self.clahe.apply(l_channel)

            # Merge back and convert to RGB
            lab_img = cv2.merge((l_channel, a_channel, b_channel))
            img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        else:
            # If the image is grayscale, apply CLAHE directly
            img = self.clahe.apply(img)

        return img
    
    
class ResNet50(nn.Module):
    def __init__(self, out_size):
        super(ResNet50, self).__init__()
        # Use the latest ImageNet weights for ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()  # Assuming you're doing a binary classification, adjust as needed
        )

    def forward(self, x):
        return self.resnet50(x)
    
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # Use the latest ImageNet weights
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x) 

def lateral_get_heatmap(image_path: str) -> Image.Image:
    """
    Generates and returns a heatmap overlay image (as a PIL Image) using Grad-CAM++.

    Args:
        image_path (str): Path to the input X-ray image.

    Returns:
        PIL.Image.Image: The Grad-CAM overlay image.
    """
    N_LABELS = 14
    # Load the saved model
    lateral_model = ResNet50(out_size=N_LABELS)
    lateral_model.load_state_dict(torch.load(r'D:\CODES\MedXRay-CAD\models\weights\lateral_best_model.pth', map_location=device))
    lateral_model = lateral_model.to(device)
    lateral_model.eval()  # Set the model to evaluation mode
    
    # Load and preprocess the image for ResNet50
    img = Image.open(image_path)
    clahe_transform = CLAHETransform(clip_limit=0.34, tile_grid_size=(8, 8))
    img = clahe_transform(img)
    img = cv2.resize(img, (256, 256))
    img = np.float32(img) / 255
    if img.ndim == 2:  # If grayscale, convert to RGB
        img = np.stack([img] * 3, axis=-1)
    input_tensor = torch.from_numpy(img[np.newaxis, ...]).permute(0, 3, 1, 2).to(device)

    # Define the target layer and targets for ResNet50
    targets = [ClassifierOutputTarget(13)]  # Replace 13 with the target class index as needed
    target_layers = [lateral_model.resnet50.layer4[-1]]

    # Use GradCAM++ to generate the visualization for ResNet50
    with GradCAMPlusPlus(model=lateral_model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    overlay_image = Image.fromarray(cam_image)
    return img, overlay_image


def frontal_get_heatmap(image_path: str) -> Image.Image:
    """
    Generates and returns a heatmap overlay image (as a PIL Image) using Grad-CAM++.

    Args:
        image_path (str): Path to the input X-ray image.

    Returns:
        PIL.Image.Image: The Grad-CAM overlay image.
    """
    N_LABELS = 14
    # Load the saved model
    best_model = DenseNet121(out_size=N_LABELS)
    best_model.load_state_dict(torch.load(r'D:\CODES\MedXRay-CAD\models\weights\frontal_best_model.pth', map_location=device))
    best_model = best_model.to(device)
    best_model.eval()  # Set the model to evaluation mode
    
    # Load and preprocess the image for DenseNet121
    img = Image.open(image_path)
    clahe_transform = CLAHETransform(clip_limit=0.34, tile_grid_size=(8, 8))
    img = clahe_transform(img)
    img = cv2.resize(img, (256, 256))
    img = np.float32(img) / 255
    img = np.stack([img] * 3, axis=-1)
    input_tensor = torch.from_numpy(img[np.newaxis, ...]).permute(0, 3, 1, 2)

    targets = [ClassifierOutputTarget(13)]
    target_layers = [best_model.densenet121.features.denseblock4]
    with GradCAMPlusPlus(model=best_model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

    overlay_image = Image.fromarray(cam_image)
    return img, overlay_image