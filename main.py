import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b0, ResNet50_Weights, EfficientNet_B0_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(EnhancedCNN, self).__init__()
        # Use EfficientNet as the base model for better efficiency
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model._fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Replace the original classifier
        self.base_model._fc = self.classifier

    def forward(self, x):
        return self.base_model(x)

class SceneSpeak:
    def __init__(self):
        # Enhanced CNN for object detection
        self.enhanced_cnn = EnhancedCNN()
        self.enhanced_cnn.eval()

        # Load BLIP model for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Define image preprocessing with data augmentation
        self.preprocess = Compose([
            Resize((256, 256)),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_file):
        """Load image from file path or URL"""
        if isinstance(image_file, str) and image_file.startswith('http'):
            return Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
        else:
            return Image.open(image_file).convert('RGB')

    def detect_objects(self, image, top_k=3):
        """
        Detect top K objects in the image with confidence scores
        
        Args:
            image (PIL.Image): Input image
            top_k (int): Number of top predictions to return
        
        Returns:
            list: Top K objects with their confidence scores
        """
        input_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = self.enhanced_cnn(input_tensor)
            probabilities = F.softmax(output, dim=1)
            top_prob, top_classes = torch.topk(probabilities, top_k)
        
        # Map class indices to labels
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        category_names = weights.meta["categories"]
        
        # Create list of detected objects with confidence
        detected_objects = [
            {
                "object": category_names[cls],
                "confidence": prob.item() * 100
            } 
            for cls, prob in zip(top_classes[0], top_prob[0])
        ]
        
        return detected_objects

    def generate_caption(self, image):
        """Generate descriptive caption for the image"""
        inputs = self.blip_processor(image, return_tensors="pt")
        output = self.blip_model.generate(**inputs, max_new_tokens=30)
        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def describe_scene(self, image_path):
        """Comprehensive scene description"""
        image = self.load_image(image_path)
        
        # Detect top objects
        detected_objects = self.detect_objects(image)
        
        # Generate caption
        caption = self.generate_caption(image)
        
        # Format description
        description = "Detected Objects:\n"
        for obj in detected_objects:
            description += f"- {obj['object']} (Confidence: {obj['confidence']:.2f}%)\n"
        description += f"\nScene Caption: {caption}"
        
        return description

# Example usage
scene_speak = SceneSpeak()
description = scene_speak.describe_scene(r"path/to/your/image.jpg")
print(description)
