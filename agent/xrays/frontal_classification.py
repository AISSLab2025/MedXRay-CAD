import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# RAG
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import csv

# Define the labels as used during training
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 
          'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

class DenseNetModel(nn.Module):
    def __init__(self, out_size=14):
        super(DenseNetModel, self).__init__()
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DenseNetModel().to(device)
model.load_state_dict(torch.load('../models/weights/frontal_best_model.pth'))
model.eval()

# CLAHE transform class
class CLAHETransform:
    def __init__(self, clip_limit=0.34, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 3:
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_img)
            l_channel = self.clahe.apply(l_channel)
            lab_img = cv2.merge((l_channel, a_channel, b_channel))
            img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        else:
            img = self.clahe.apply(img)
        return Image.fromarray(img.astype('uint8'))

# Define the validation transform (same as during training)
transform = transforms.Compose([
    transforms.Resize(256),
    CLAHETransform(clip_limit=0.35, tile_grid_size=(8, 8)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Function to predict an image
def predict_image_frontal(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model(image)
    
    # Convert predictions to numpy array and map to labels
    predictions = output.cpu().numpy().squeeze()  # Remove batch dimension
    pred_scores = {labels[i]: predictions[i] for i in range(len(predictions))}
    
    return pred_scores

# # Example usage
# image_path = r"D:\CODES\MICCAI2025\images\p11121324_s58365087.jpg" # Real labels for this image {Consolidation, Pleural Effusion, Supporting Device}
# pred_scores = predict_image_frontal(image_path)
# # pred_scores
# # Print the scores for each label
# for label, score in pred_scores.items():
#     print(f'{label}: {score:.4f}')