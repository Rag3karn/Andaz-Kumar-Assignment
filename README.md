# 3D Model Generator from Images/Text

This project generates 3D models (.obj/.stl) from either images or text prompts using AI/ML techniques.

## Features

- Image to 3D model conversion
- Text to 3D model generation
- Background removal for image inputs
- 3D model visualization
- Export to .obj/.stl formats

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. For image input:
```bash
python main.py --input path/to/image.jpg --mode image
```

2. For text input:
```bash
python main.py --input "A small toy car" --mode text
```

## Project Structure

- `main.py`: Main entry point
- `image_processor.py`: Image preprocessing and background removal
- `model_generator.py`: 3D model generation logic
- `visualizer.py`: 3D model visualization
- `utils.py`: Utility functions

## Libraries Used

- OpenCV: Image processing
- Rembg: Background removal
- PyTorch: Deep learning framework
- Transformers: Text-to-3D model generation
- PyRender: 3D visualization
- Trimesh: 3D model manipulation

## Thought Process

1. Image Processing Pipeline:
   - Input image preprocessing
   - Background removal using Rembg
   - Feature extraction

2. Text-to-3D Pipeline:
   - Text prompt processing
   - Using pre-trained models for 3D generation
   - Post-processing for better quality

3. 3D Model Generation:
   - Converting features to 3D mesh
   - Optimizing mesh topology
   - Exporting to standard formats

## License

MIT License