import pyrender
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ModelVisualizer:
    def __init__(self):
        self.scene = pyrender.Scene()

    def visualize_mesh(self, mesh, output_path=None):
        """Visualize 3D mesh using pyrender."""
        try:
            # Convert trimesh to pyrender mesh
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8
            )
            
            # Create pyrender mesh
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            # Add mesh to scene
            self.scene.add(pyrender_mesh)
            
            # Set up camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            self.scene.add(camera, pose=camera_pose)
            
            # Set up light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            light_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            self.scene.add(light, pose=light_pose)
            
            # Render scene
            r = pyrender.OffscreenRenderer(800, 600)
            color, depth = r.render(self.scene)
            
            # Save visualization if output path is provided
            if output_path:
                plt.imsave(output_path, color)
            
            return color
            
        except Exception as e:
            raise Exception(f"Error visualizing mesh: {str(e)}")
        finally:
            # Clean up
            self.scene.clear()

    def plot_mesh(self, mesh, output_path=None):
        """Create a simple matplotlib visualization of the mesh."""
        try:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot vertices
            vertices = np.array(mesh.vertices)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.')
            
            # Plot faces
            for face in mesh.faces:
                vertices_face = vertices[face]
                ax.plot_trisurf(vertices_face[:, 0], vertices_face[:, 1], vertices_face[:, 2],
                              color='gray', alpha=0.3)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Model Visualization')
            
            # Save plot if output path is provided
            if output_path:
                plt.savefig(output_path)
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting mesh: {str(e)}")
        finally:
            plt.close() 