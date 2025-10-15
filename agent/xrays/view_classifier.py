import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import densenet121
import torch.nn as nn

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        self.densenet = densenet121(pretrained=False)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)

def preprocess_image(image_path, output_size=(224, 224)):
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    return input_batch

def view_classifier(image_path):
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = DenseNetModel().to(device)
    model.load_state_dict(torch.load('../models/weights/view_classifier.pt', map_location=device))
    model.eval()
    
    # Preprocess the image
    input_batch = preprocess_image(image_path)
    input_batch = input_batch.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        
    # Get the predicted class and confidence
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    # Map class index to label
    class_labels = ['frontal', 'lateral']
    result = class_labels[predicted_class]
    
    return result