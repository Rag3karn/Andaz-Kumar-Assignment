import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import trimesh
from scipy.spatial import Delaunay
import cv2

class ModelGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

    def generate_from_text(self, text_prompt):
        """Generate 3D model from text prompt."""
        try:
            # Generate image from text using Stable Diffusion
            image = self.diffusion_pipeline(text_prompt).images[0]
            
            # Convert image to numpy array
            image_np = np.array(image)
            
            # Generate depth map (simplified version)
            depth_map = self._generate_depth_map(image_np)
            
            # Convert depth map to 3D mesh
            mesh = self._depth_to_mesh(depth_map)
            
            return mesh
        except Exception as e:
            raise Exception(f"Error generating 3D model from text: {str(e)}")

    def generate_from_image(self, image_tensor):
        """Generate 3D model from processed image tensor."""
        try:
            # Convert tensor to numpy array
            image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
            
            # Generate depth map
            depth_map = self._generate_depth_map(image_np)
            
            # Convert depth map to 3D mesh
            mesh = self._depth_to_mesh(depth_map)
            
            return mesh
        except Exception as e:
            raise Exception(f"Error generating 3D model from image: {str(e)}")

    def _generate_depth_map(self, image):
        """Generate a simplified depth map from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Generate depth map using Laplacian
        depth_map = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map

    def _depth_to_mesh(self, depth_map):
        """Convert depth map to 3D mesh."""
        height, width = depth_map.shape
        
        # Create grid of points
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Stack coordinates
        points = np.stack([x, y, depth_map], axis=-1)
        
        # Create triangulation
        tri = Delaunay(points[:, :, :2])
        
        # Create mesh
        vertices = points.reshape(-1, 3)
        faces = tri.simplices
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Clean up mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        
        return mesh

    def save_mesh(self, mesh, output_path, format='obj'):
        """Save mesh to file."""
        try:
            if format.lower() == 'obj':
                mesh.export(output_path, file_type='obj')
            elif format.lower() == 'stl':
                mesh.export(output_path, file_type='stl')
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise Exception(f"Error saving mesh: {str(e)}") 