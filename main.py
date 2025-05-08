import argparse
import os
import matplotlib.pyplot as plt
from image_processor import ImageProcessor
from model_generator import ModelGenerator
from visualizer import ModelVisualizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate 3D models from images or text')
    parser.add_argument('--input', required=True, help='Input image path or text prompt')
    parser.add_argument('--mode', required=True, choices=['image', 'text'], help='Input mode: image or text')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--format', default='obj', choices=['obj', 'stl'], help='Output format')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    try:
        # Initialize components
        image_processor = ImageProcessor()
        model_generator = ModelGenerator()
        visualizer = ModelVisualizer()

        # Process input based on mode
        if args.mode == 'image':
            # Process image
            processed_data = image_processor.process_image(args.input)
            
            # Generate 3D model
            mesh = model_generator.generate_from_image(processed_data['processed'])
            
            # Save original image without background
            image_path = os.path.join(args.output, 'processed_image.png')
            plt.imsave(image_path, processed_data['no_background'])
            
        else:  # text mode
            # Generate 3D model from text
            mesh = model_generator.generate_from_text(args.input)

        # Save 3D model
        model_path = os.path.join(args.output, f'model.{args.format}')
        model_generator.save_mesh(mesh, model_path, format=args.format)

        # Generate visualizations
        render_path = os.path.join(args.output, 'render.png')
        plot_path = os.path.join(args.output, 'plot.png')
        
        # Create both types of visualizations
        visualizer.visualize_mesh(mesh, output_path=render_path)
        visualizer.plot_mesh(mesh, output_path=plot_path)

        print(f"Successfully generated 3D model and visualizations in {args.output}")
        print(f"Model saved as: {model_path}")
        print(f"Visualizations saved as: {render_path} and {plot_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main()) 