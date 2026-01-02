# 🏗️ Deep Learning Architectures - Complete Guide

## Neural Network Fundamentals

### Feedforward Neural Networks (FNN)
- Basic architecture: Input → Hidden Layers → Output
- Activation functions: ReLU, Sigmoid, Tanh
- Applications: Tabular data, general purpose

### Convolutional Neural Networks (CNN)
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Architecture: Conv → Activation → Pooling → Dense
- Applications: Image classification, object detection

---

## Advanced Architectures

### 1. Residual Networks (ResNet)
- Skip connections to enable deep networks
- Addresses vanishing gradient problem
- Variants: ResNet18, ResNet50, ResNet152

### 2. Vision Transformers (ViT)
- Transformer architecture for images
- Patch-based approach
- Competitive with CNNs on ImageNet

### 3. Transformers (NLP)
- Self-attention mechanism
- Attention is All You Need (2017)
- Models: BERT, GPT, T5, ALBERT

### 4. Recurrent Neural Networks (RNN)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional variants
- Applications: Time series, sequence modeling

### 5. Graph Neural Networks (GNN)
- Graph Convolutional Networks (GCN)
- GraphSAGE, GAT (Graph Attention Networks)
- Applications: Social networks, molecular graphs

### 6. Generative Models
- **GANs**: Generator & Discriminator
- **VAE**: Variational Autoencoders
- **Diffusion Models**: DALL-E 3, Stable Diffusion
- **Flow Models**: Normalizing flows

---

## Popular Pre-trained Models

### Computer Vision
- EfficientNet, MobileNet (efficient models)
- Inception, DenseNet
- YOLO (object detection)
- Mask R-CNN (instance segmentation)

### Natural Language Processing
- BERT, RoBERTa, ALBERT (encoders)
- GPT-2, GPT-3, GPT-4 (decoders)
- T5 (encoder-decoder)
- ELECTRA, XLNet (advanced encoders)

### Multi-Modal
- CLIP (vision + language)
- DALL-E (text → image)
- Flamingo (vision-language understanding)

---

## Training Techniques

### Optimization
- SGD, Adam, RMSprop, AdamW
- Learning rate scheduling
- Gradient clipping

### Regularization
- Dropout, Batch Normalization
- L1/L2 regularization
- Data augmentation
- Early stopping

### Advanced Training
- Mixed precision training
- Gradient accumulation
- Distributed training (DDP, FSDP)

---

## Architecture Patterns

```
Image Classification Pipeline:
Input Image → CNN (FeatureExtraction) → Flatten → Dense → Softmax → Class Probabilities

Object Detection Pipeline:
Input Image → CNN (Backbone) → RPN (Region Proposals) → Classification & Localization

NLP Pipeline:
Text → Tokenization → Embeddings → Transformer → Pooling → Classification

Time Series Pipeline:
Sequential Data → LSTM/Transformer → Attention → Dense → Prediction
```

---

## 📊 Model Comparison

| Model | Task | Accuracy | Speed | Memory |
|-------|------|----------|-------|--------|
| ResNet50 | ImageNet | 76% | Fast | Medium |
| Vision Transformer | ImageNet | 88% | Medium | High |
| BERT | NLP | High | Slow | High |
| GPT-4 | Language | SOTA | Slow | Very High |
| EfficientNet | ImageNet | 85% | Very Fast | Low |

---

*Detailed implementations and code examples available in projects folder.*
