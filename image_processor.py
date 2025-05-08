import cv2
import numpy as np
from rembg import remove
from PIL import Image
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        """Load and preprocess the input image."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

    def remove_background(self, image):
        """Remove background from the image using rembg."""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Remove background
            output = remove(pil_image)
            
            # Convert back to numpy array
            return np.array(output)
        except Exception as e:
            raise Exception(f"Error removing background: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply transformations
            tensor = self.transform(pil_image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def process_image(self, image_path):
        """Complete image processing pipeline."""
        # Load image
        image = self.load_image(image_path)
        
        # Remove background
        image_no_bg = self.remove_background(image)
        
        # Preprocess for model
        processed_tensor = self.preprocess_image(image_no_bg)
        
        return {
            'original': image,
            'no_background': image_no_bg,
            'processed': processed_tensor
        } 