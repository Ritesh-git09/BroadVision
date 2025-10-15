# 🎨 BroadVision - Advanced 2D to 3D Conversion

Transform 2D images into high-quality 3D models using state-of-the-art deep learning models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 🌟 Features

- **Advanced Single-Image 3D Reconstruction** with ZoeDepth/DPT
- **Multi-View Stereo Reconstruction** for maximum accuracy
- **Multiple Mesh Generation Algorithms** (Poisson, Ball-Pivoting, Alpha Shape)
- **Automated Point Cloud Cleaning** and optimization
- **Multiple Export Formats** (.obj, .ply, .stl) for Unity, Blender, and 3D printing

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- At least 4GB RAM (8GB+ recommended)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BroadVision.git
cd BroadVision
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/Mac

# Install required packages
pip install -r requirements_advanced.txt
```

### 3. Download AI Models

**The AI models will be automatically downloaded on first run**, but you can also pre-download them:

#### Option A: Automatic Download (Recommended)
Just run the script - models will download automatically:
```bash
python advanced_image_to_3d.py
```

#### Option B: Manual Pre-download
If you want to download models in advance:

```python
# Run this Python script to pre-download models
python -c "
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Download DPT-Large (primary model)
print('Downloading DPT-Large model...')
DPTFeatureExtractor.from_pretrained('Intel/dpt-large')
DPTForDepthEstimation.from_pretrained('Intel/dpt-large')

# Download ZoeDepth (best quality - optional)
print('Downloading ZoeDepth model...')
try:
    model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_NK', pretrained=True)
    print('✓ All models downloaded successfully!')
except Exception as e:
    print(f'ZoeDepth download failed (will use DPT): {e}')
"
```

**Model Storage Locations:**
- DPT models: `~/.cache/huggingface/` (~1.4 GB)
- ZoeDepth: `~/.cache/torch/hub/` (~1.35 GB)
- Total space needed: ~3 GB

### 4. Prepare Your Image

Place your image in the project folder:
```bash
# Copy your image
cp /path/to/your/image.jpg test.jpg
```

### 5. Run the Conversion

**For Single Image:**
```bash
python advanced_image_to_3d.py
```

**For Multiple Images (Best Quality):**
```bash
# First, take 3-5 photos of your object from different angles
# Name them: view1.jpg, view2.jpg, view3.jpg, etc.
python multiview_3d.py
```

## 🎯 Usage Examples

### Basic Usage - Single Image

```python
from advanced_image_to_3d import AdvancedImage3D

# Create processor
processor = AdvancedImage3D()

# Process image with default settings
mesh, pcd = processor.process(
    image_path="test.jpg",
    output_name="my_model"
)
```

### Advanced Usage - Custom Parameters

```python
processor = AdvancedImage3D()

mesh, pcd = processor.process(
    image_path="test.jpg",
    output_name="my_model",
    depth_scale=3.0,          # Increase depth (1.0-5.0)
    mesh_method='poisson',    # or 'bpa', 'alpha'
    smooth_iterations=5,      # More smoothing
    visualize=True            # Show 3D viewer
)
```

### Multi-View Reconstruction

```python
from multiview_3d import MultiView3D

# Use multiple images for better quality
reconstructor = MultiView3D()
mesh, pcd = reconstructor.process(
    image_paths=["view1.jpg", "view2.jpg", "view3.jpg"],
    output_name="multiview_model"
)
```

## ⚙️ Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `DEPTH_SCALE` | 2.0 | 0.5-5.0 | Controls depth intensity |
| `MESH_METHOD` | 'poisson' | poisson/bpa/alpha | Mesh reconstruction algorithm |
| `SMOOTH_ITER` | 3 | 0-10 | Smoothing iterations |
| `VISUALIZE` | True | True/False | Show 3D visualization |

### Tuning Tips

- **Too flat?** → Increase `DEPTH_SCALE` to 3.0-5.0
- **Too distorted?** → Decrease `DEPTH_SCALE` to 1.0-1.5
- **Too rough?** → Increase `SMOOTH_ITER` to 5-10
- **Missing details?** → Try different `MESH_METHOD` values

## 📊 Method Comparison

| Method | Quality | Speed | Requirements | Use Case |
|--------|---------|-------|--------------|----------|
| **Original MiDaS** | ⭐⭐ | Fast | 1 image | Quick tests |
| **Advanced DPT** | ⭐⭐⭐⭐ | Medium | 1 image | Best single-image |
| **ZoeDepth** | ⭐⭐⭐⭐⭐ | Medium | 1 image | Metric depth needed |
| **Multi-View** | ⭐⭐⭐⭐⭐ | Slow | 2+ images | Maximum accuracy |

## 🎨 Output Files

After processing, you'll get:

```
advanced_output.obj          # Mesh for Unity/Blender (with colors)
advanced_output.ply          # High-quality mesh format
advanced_output.stl          # For 3D printing
advanced_output_pointcloud.ply  # Raw point cloud data
```

## 🖼️ Best Practices

### For Single-Image Reconstruction:
- ✅ Use high-resolution images (1080p+)
- ✅ Good lighting, minimal shadows
- ✅ Objects with clear depth cues
- ✅ Textured surfaces work better than flat
- ❌ Avoid motion blur or noise

### For Multi-View Reconstruction:
- ✅ Take 3-10 photos from different angles
- ✅ Maintain 70% overlap between views
- ✅ Keep ~15-30° spacing between shots
- ✅ Use consistent lighting
- ✅ Include distinctive features/textures
- ❌ Avoid reflective or transparent objects

## 🔧 Troubleshooting

### Models Not Downloading?

```bash
# Check internet connection and retry
pip install --upgrade huggingface_hub torch

# Manual download
python -c "from transformers import DPTForDepthEstimation; DPTForDepthEstimation.from_pretrained('Intel/dpt-large')"
```

### Out of Memory?

```python
# Use CPU instead of GPU
processor = AdvancedImage3D(device='cpu')
```

### Poor Quality Results?

1. Try adjusting `DEPTH_SCALE` parameter
2. Use higher resolution input image
3. Try different `MESH_METHOD` values
4. Consider using multi-view approach

### Visualization Not Opening?

```python
# Disable visualization and just save files
processor.process(..., visualize=False)
```

## 📚 Project Structure

```
BroadVision/
├── advanced_image_to_3d.py    # Main single-image script
├── multiview_3d.py            # Multi-view reconstruction
├── image_to_3d.py             # Original basic version
├── requirements_advanced.txt  # Dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
└── test.jpg                   # Your input image (not in repo)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [ZoeDepth](https://github.com/isl-org/ZoeDepth) - Metric depth estimation
- [DPT](https://github.com/isl-org/DPT) - Vision Transformers for dense prediction
- [Open3D](http://www.open3d.org/) - 3D data processing
- [Hugging Face](https://huggingface.co/) - Model hosting

## 📞 Support

If you encounter any issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/BroadVision/issues)
3. Create a new issue with details about your problem

## 🔮 Future Enhancements

- [ ] NeRF integration for view synthesis
- [ ] Real-time processing
- [ ] GUI interface
- [ ] Texture UV mapping
- [ ] Batch processing support
- [ ] Cloud processing API

## ⚡ Performance Tips

- Use GPU for 3-5x faster processing
- Process images in batch for multiple objects
- Reduce image resolution for faster (but lower quality) results
- Use `VISUALIZE=False` when processing many images

## 🎓 Learn More

- [Understanding Depth Estimation](https://arxiv.org/abs/2302.12288)
- [3D Reconstruction Techniques](http://www.open3d.org/docs/latest/tutorial/reconstruction_system/index.html)
- [Photogrammetry Basics](https://en.wikipedia.org/wiki/Photogrammetry)

---

**Made with ❤️ by BroadVision Team**

*Star ⭐ this repo if you found it helpful!*
