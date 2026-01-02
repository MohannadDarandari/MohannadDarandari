# 🖼️ Image Processing & Advanced Topics

## Overview

Image processing is the manipulation and analysis of digital images using computational techniques.

---

## 🎨 Fundamental Operations

### Point Operations (Pixel-Level)
- **Brightness/Contrast**: Multiply or add constant to all pixels
- **Gamma Correction**: Adjust image brightness non-linearly
- **Threshold**: Convert to binary (black/white)
- **Inverted**: 255 - pixel value
- **Clipping**: Bound pixel values

### Spatial Filtering

#### Convolution
- **Process**: Slide kernel over image, compute weighted sum
- **Kernel**: Small matrix (3×3, 5×5, etc.)
- **Applications**: Blurring, edge detection, sharpening
- **Properties**: Commutative, associative, distributive

#### Common Kernels
- **Gaussian Blur**: Smooth image, reduce noise
- **Sobel**: Detect edges
- **Laplacian**: Edges & texture
- **Median Filter**: Remove salt-and-pepper noise
- **Bilateral Filter**: Edge-preserving blur

### Morphological Operations
- **Erosion**: Remove small objects
- **Dilation**: Fill small holes
- **Opening**: Erosion then dilation
- **Closing**: Dilation then erosion
- **Skeletonization**: Thin structures to single pixel

---

## 🔍 Feature Detection

### Edge Detection
- **Sobel**: Gradient-based (x & y)
- **Canny**: Multi-stage (gradients, non-max suppression, hysteresis)
- **Laplacian of Gaussian (LoG)**: Scale-space edge detection
- **Prewitt**: Similar to Sobel
- **Roberts**: Simple 2×2 kernels

### Corner Detection
- **Harris Corner**: Intensity changes in all directions
- **Shi-Tomasi**: Modified Harris
- **FAST**: Fast segment test
- **BRIEF**: Binary descriptors

### Scale-Invariant Features
- **SIFT**: Scale-Invariant Feature Transform
  - Detects keypoints across scales
  - 128-dim descriptor
  - Rotation invariant
  
- **SURF**: Speeded-Up Robust Features
  - Faster than SIFT
  - Integral images
  - 64-dim descriptor

- **ORB**: Oriented FAST and Rotated BRIEF
  - Fast, efficient
  - Open-source (unlike SIFT)
  - Good for real-time

- **AKAZE**: Accelerated-KAZE
  - Scale-space extrema
  - Fast on mobile

---

## 🎭 Image Segmentation

### Semantic Segmentation
- **FCN**: Fully Convolutional Networks
- **U-Net**: Encoder-decoder with skip connections
- **DeepLab**: Atrous convolution & CRF
- **SegNet**: Efficient segmentation
- **Mask2Former**: State-of-the-art

### Instance Segmentation
- **Mask R-CNN**: Object + mask prediction
- **YOLACT**: Real-time instance segmentation
- **CondInst**: Conditional instance normalization

### Panoptic Segmentation
- **Stuff**: Background classes (road, sky)
- **Things**: Object classes (person, car)
- **Combined**: Stuff + Things in one prediction

---

## 🎥 Video Processing

### Optical Flow
- **Lucas-Kanade**: Local motion estimation
- **Horn-Schunck**: Global smoothness constraint
- **FlowNet**: Deep learning approach
- **PWCNet**: Pyramid, warping, cost volume

### Video Stabilization
- **Motion estimation**: Compute camera motion
- **Motion compensation**: Adjust frames
- **Temporal smoothing**: Smooth motion path

### Video Object Tracking
- **Kalman Filter**: Predict & correct tracking
- **Particle Filter**: Multiple hypothesis tracking
- **Deep SORT**: Deep learning + Kalman + Hungarian

### Action Recognition
- **3D CNN**: Spatial-temporal convolution
- **Two-Stream**: Appearance + optical flow
- **Temporal Segment Networks**: Long-range patterns

---

## 📐 Geometric Transformations

### Affine Transformations
- **Rotation**: Rotate by angle θ
- **Translation**: Shift by (tx, ty)
- **Scaling**: Zoom in/out
- **Shear**: Skew transformation
- **Composition**: Combine multiple transforms

### Perspective Transformation
- **Homography**: 8-parameter transformation
- **4-point correspondence**: Define transformation
- **Applications**: Document scanning, aerial view

### Image Registration
- **Feature-based**: Match keypoints
- **Intensity-based**: Optimize pixel similarity
- **Optical flow-based**: Use motion estimation

---

## 🎨 Color Processing

### Color Spaces
- **RGB**: Red, Green, Blue (additive)
- **HSV**: Hue, Saturation, Value (perceptual)
- **LAB**: Lightness, a (red-green), b (yellow-blue)
- **YCbCr**: Luma, chrominance (video standard)
- **Grayscale**: Single channel (luminosity)

### Color Enhancement
- **White Balance**: Correct color temperature
- **Histogram Equalization**: Improve contrast
- **Color Grading**: Artistic color correction
- **Inpainting**: Fill missing regions

---

## 🔬 Advanced Techniques

### Texture Analysis
- **Gabor Filters**: Orientation & frequency
- **Local Binary Patterns (LBP)**: Local texture descriptor
- **Haralick Features**: Texture from co-occurrence matrix
- **Wavelet Analysis**: Multi-scale decomposition

### Image Restoration
- **Denoising**: Remove noise while preserving edges
  - Gaussian blur (simple)
  - Bilateral filter (edge-preserving)
  - Non-local means (patch-based)
  - Deep learning (denoising autoencoder)

- **Deblurring**: Reverse motion or defocus blur
  - Wiener filter
  - Richardson-Lucy
  - Deep learning

- **Super-Resolution**: Increase resolution
  - Interpolation (nearest, bilinear, bicubic)
  - SRCNN: Super-Resolution CNN
  - ESPCN: Efficient Sub-Pixel
  - SRResNet: Residual networks

### Image Compression
- **JPEG**: Discrete Cosine Transform (DCT)
- **PNG**: Lossless compression
- **Wavelet Compression**: Multi-scale decomposition
- **Neural Image Compression**: Deep learning approach

---

## 🏥 Medical Image Processing

### Modalities
- **CT**: Computed Tomography (X-ray slices)
- **MRI**: Magnetic Resonance Imaging
- **Ultrasound**: Sound wave reflections
- **PET**: Positron Emission Tomography
- **X-ray**: Radiography

### Tasks
- **Registration**: Align images over time
- **Segmentation**: Identify organs/tumors
- **Classification**: Disease detection
- **Reconstruction**: 3D from 2D slices
- **Quality Enhancement**: Reduce noise/artifacts

### Challenges
- **3D data**: Large volume
- **Small datasets**: Limited training data
- **Privacy**: HIPAA compliance
- **Interpretability**: Clinical requirements

---

## 🧬 3D Vision

### Depth Estimation
- **Stereo**: Two calibrated cameras
- **Structure from Motion**: 3D from image sequence
- **Photometric Stereo**: Multiple lighting conditions
- **Monocular Depth**: Single image deep learning

### 3D Reconstruction
- **Voxel-based**: 3D grid of occupancy
- **Mesh-based**: Surface representation
- **Point Cloud**: 3D point set
- **Implicit Surfaces**: Neural representations

### 3D Object Detection
- **YOLO 3D**: Bounding box in 3D space
- **PointNet**: Direct point cloud processing
- **PointNet++**: Hierarchical point features

---

## 📺 Real-Time Processing

### Optimization Techniques
- **Resolution Reduction**: Lower input resolution
- **Model Quantization**: Reduce precision (int8, fp16)
- **Model Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Train smaller model
- **Mobile Architectures**: MobileNet, SqueezeNet

### Hardware Acceleration
- **GPU**: CUDA, OpenGL
- **TPU**: Tensor Processing Unit
- **FPGA**: Reconfigurable hardware
- **Edge AI**: On-device processing

---

## 📚 Tools & Libraries

### OpenCV
- Computer vision library (C++, Python)
- 2500+ algorithms
- Real-time processing

### PIL/Pillow
- Python Image Library
- Image manipulation
- Format conversion

### scikit-image
- Scientific image processing
- Filters, restoration, segmentation

### TensorFlow/PyTorch
- Deep learning for vision
- Pre-trained models
- Custom architectures

### MediaPipe
- Ready-to-use solutions
- Pose, hand, face detection
- Mobile-optimized

---

*Detailed implementations in projects folder.*
