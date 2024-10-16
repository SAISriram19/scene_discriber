import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneSpeak:
    def __init__(self):
        # Load pre-trained ResNet model for object detection
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.eval()

        # Load BLIP model for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Define image preprocessing
        self.preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_file):
        if isinstance(image_file, str) and image_file.startswith('http'):
            return Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
        else:
            return Image.open(image_file).convert('RGB')  # Use the UploadedFile object directly


    def detect_objects(self, image):
        input_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = self.resnet(input_tensor)
        _, predicted = torch.max(output, 1)
        return ResNet50_Weights.IMAGENET1K_V2.meta["categories"][predicted.item()]

    def generate_caption(self, image):
        inputs = self.blip_processor(image, return_tensors="pt")
        output = self.blip_model.generate(**inputs, max_new_tokens=20)
        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def describe_scene(self, image_path):
        image = self.load_image(image_path)
        main_object = self.detect_objects(image)
        caption = self.generate_caption(image)
        return f"Main object detected: {main_object}\nScene description: {caption}"

# Example usage
scene_speak = SceneSpeak()
description = scene_speak.describe_scene(r"C:\Users\saisr\OneDrive\Pictures\YST01427.JPG")
print(description)