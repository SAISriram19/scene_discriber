import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# Reduce logging overhead
logging.basicConfig(level=logging.ERROR)  # Only log critical errors

class SceneSpeak:
    _instance = None
    
    def __new__(cls):
        # Singleton pattern for resource efficiency
        if not cls._instance:
            cls._instance = super(SceneSpeak, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Efficient model loading with minimal overhead
        self.enhanced_cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(self.device)
        self.enhanced_cnn.eval()
        
        # Use half precision for memory efficiency
        if self.device.type == 'cuda':
            self.enhanced_cnn = self.enhanced_cnn.half()
        
        # Lazy load BLIP model to reduce initial startup time
        self.blip_processor = None
        self.blip_model = None
        
        # Cached weights for faster reference
        self._category_names = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
        
        # Efficient preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # Fixed size for consistent processing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._initialized = True
    
    def _lazy_load_blip(self):
        """Lazy load BLIP model when first needed"""
        if self.blip_processor is None:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
    
    def load_image(self, image_file):
        """Efficient image loading with minimal overhead"""
        try:
            image = Image.open(
                requests.get(image_file, stream=True).raw if image_file.startswith('http') 
                else image_file
            ).convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Image load error: {e}")
            raise

    def detect_objects(self, image, top_k=3):
        """Optimized object detection"""
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Use half precision if on GPU
        if self.device.type == 'cuda':
            input_tensor = input_tensor.half()
        
        with torch.no_grad():
            output = self.enhanced_cnn(input_tensor)
            probabilities = F.softmax(output, dim=1)
            top_prob, top_classes = torch.topk(probabilities, top_k)
        
        # Efficient object extraction
        return [
            {
                "object": self._category_names[cls.item()],
                "confidence": prob.item() * 100
            } 
            for cls, prob in zip(top_classes[0], top_prob[0])
        ]

    def generate_caption(self, image):
        """Efficient caption generation with lazy loading"""
        self._lazy_load_blip()
        
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        output = self.blip_model.generate(**inputs, max_new_tokens=30)
        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def describe_scene(self, image):
        """Comprehensive and efficient scene description"""
        # Handle string input efficiently
        if isinstance(image, str):
            image = self.load_image(image)
        
        detected_objects = self.detect_objects(image)
        caption = self.generate_caption(image)
        
        return {
            "objects": detected_objects,
            "caption": caption
        }
