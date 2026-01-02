# 👁️ Computer Vision - Advanced Topics

## Fundamental Concepts

### Image Basics
- Color spaces: RGB, HSV, LAB
- Filtering: Convolution, Sobel, Canny
- Morphological operations: Erosion, Dilation, Opening, Closing

### Feature Detection
- SIFT: Scale-Invariant Feature Transform
- SURF: Speeded Up Robust Features
- ORB: Oriented FAST and Rotated BRIEF
- BRIEF: Binary Robust Independent Elementary Features

---

## Deep Learning for Computer Vision

### Convolutional Operations
- Convolution: Feature extraction
- Pooling: Downsampling (Max, Average)
- Stride & Padding
- Receptive field

### CNN Architectures

#### Classic Architectures
- **AlexNet**: ImageNet breakthrough (2012)
- **VGGNet**: Deep simple architecture
- **GoogleNet/Inception**: Multi-scale feature extraction
- **ResNet**: Skip connections for very deep networks
- **DenseNet**: Dense skip connections

#### Efficient Architectures
- **MobileNet**: Efficient mobile deployment
- **SqueezeNet**: Compact networks
- **EfficientNet**: Scaling networks systematically
- **ShuffleNet**: Channel shuffling for efficiency

### Modern Architectures
- **Vision Transformers (ViT)**: Transformer for images
- **Swin Transformers**: Hierarchical vision transformers
- **ConvNeXt**: Modernized CNNs inspired by ViTs

---

## Object Detection

### Traditional Methods
- Sliding window approaches
- HOG (Histogram of Oriented Gradients)
- Selective search

### Deep Learning Methods

#### Two-Stage Detectors
- **R-CNN**: Region-based CNN
- **Fast R-CNN**: Improved R-CNN
- **Faster R-CNN**: Region Proposal Network (RPN)
- **Mask R-CNN**: Instance segmentation + detection

#### One-Stage Detectors
- **YOLO**: You Only Look Once (v3, v4, v5, v8)
- **SSD**: Single Shot MultiBox Detector
- **RetinaNet**: Focal loss for class imbalance

#### Anchor-Free Detectors
- **CenterNet**: Center-based detection
- **FCOS**: Fully Convolutional One-Stage
- **CornerNet**: Detecting objects as paired keypoints

---

## Semantic & Instance Segmentation

### Semantic Segmentation
- **FCN**: Fully Convolutional Networks
- **U-Net**: Encoder-decoder with skip connections
- **DeepLab**: Atrous convolution & CRF
- **SegNet**: Encoder-decoder architecture

### Instance Segmentation
- **Mask R-CNN**: R-CNN + mask branch
- **YOLACT**: Real-time instance segmentation
- **CondInst**: Conditional instance normalization

### Panoptic Segmentation
- Combines semantic + instance segmentation
- **DETR-Panoptic**: Transformer-based approach

---

## Specialized Tasks

### Pose Estimation
- OpenPose: Multi-person pose estimation
- PoseNet: Lightweight pose detection
- MediaPipe: Real-time human pose
- Key points: Joints and bone connections

### Face Detection & Recognition
- **Face Detection**: MTCNN, RetinaFace, YOLOv8-face
- **Face Recognition**: FaceNet, VGGFace2, ArcFace
- **Facial Attributes**: Age, gender, expression detection
- **Face Alignment**: Landmark detection

### 3D Vision
- **Depth Estimation**: Monocular, stereo
- **3D Object Detection**: 3D bounding boxes
- **Point Cloud Processing**: PointNet, PointNet++
- **Structure from Motion**: 3D reconstruction

### Medical Imaging
- **CT/MRI Analysis**: Tumor detection
- **X-ray Classification**: Pathology detection
- **Retinal Imaging**: Diabetic retinopathy detection
- **3D Medical Imaging**: Volumetric data processing

### Video Understanding
- **Action Recognition**: Activity detection in videos
- **Video Classification**: Category prediction
- **Temporal Segmentation**: Action localization
- **3D Convolutions**: Spatial-temporal features

---

## Image Generation

### Generative Adversarial Networks (GANs)
- Generator vs Discriminator
- Loss functions: Wasserstein, Hinge
- Applications: Image synthesis, style transfer

### Diffusion Models
- **DDPM**: Denoising Diffusion Probabilistic Models
- **DDIM**: Accelerated sampling
- **Stable Diffusion**: Text-to-image generation
- **DALL-E 3**: Advanced text-to-image

### Variational Autoencoders (VAE)
- Encoder-decoder with latent space
- Applications: Image generation, interpolation

### Neural Style Transfer
- Feature visualization
- Perceptual losses
- Real-time style transfer

---

## Transfer Learning & Pre-training

### ImageNet Pre-trained Models
- Models trained on 1.2M images, 1000 classes
- Fine-tune on custom datasets
- Available: PyTorch, TensorFlow, ONNX

### Domain Adaptation
- Adversarial training
- Unsupervised adaptation
- Few-shot fine-tuning

### Self-Supervised Learning
- Contrastive learning: SimCLR, MoCo
- Masked image modeling: BEI, MAE
- Clustering-based: DeepCluster, SwAV

---

## Best Practices

### Data Preparation
- Image normalization (ImageNet stats)
- Data augmentation (rotation, flip, color jitter)
- Handle class imbalance

### Model Selection
- Start with pre-trained models
- Choose based on speed/accuracy trade-off
- Consider deployment constraints

### Training Strategy
- Use appropriate learning rates
- Batch normalization for stability
- Gradient clipping for large models
- Early stopping

### Deployment
- Model quantization for edge
- ONNX format for cross-platform
- Real-time optimization

---

## Tools & Libraries

- **OpenCV**: Traditional CV & image processing
- **scikit-image**: Scientific image processing
- **Pillow**: Python Imaging Library
- **PyTorch Vision**: Vision models & datasets
- **TensorFlow Hub**: Pre-trained models
- **MediaPipe**: Ready-to-use solutions

---

*Detailed code examples in projects folder.*
