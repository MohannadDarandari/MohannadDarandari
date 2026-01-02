# 📚 Complete AI/ML Libraries & Tools Reference

## Deep Learning Frameworks

### PyTorch
- **Version**: 2.1.0+
- **Use**: Production ML, Research
- **Strengths**: Dynamic computation graphs, flexible, research-friendly
- **Key Libraries**: TorchVision, TorchText, TorchAudio, PyTorch Lightning
- **Installation**: `pip install torch torchvision`

### TensorFlow/Keras
- **Version**: 2.13.0+
- **Use**: Enterprise ML, Production
- **Strengths**: Production-ready, ecosystem, deployment tools
- **Key Components**: Keras API, TF Lite, TF Serving, TFX
- **Installation**: `pip install tensorflow`

### JAX
- **Version**: 0.4.0+
- **Use**: Research, Scientific Computing
- **Strengths**: Functional programming, JIT compilation, autodiff
- **Installation**: `pip install jax jaxlib`

### Hugging Face Transformers
- **Version**: 4.30.0+
- **Use**: NLP, Vision, Multi-modal
- **Models**: BERT, GPT, T5, CLIP, ViT, 50K+
- **Installation**: `pip install transformers`

---

## Machine Learning Libraries

### Scikit-learn
- **Version**: 1.3.0+
- **Use**: Classical ML algorithms
- **Coverage**: Classification, regression, clustering, preprocessing
- **Installation**: `pip install scikit-learn`

### XGBoost / LightGBM / CatBoost
- **Version**: Latest stable
- **Use**: Gradient boosting models
- **Strengths**: SOTA performance on tabular data
- **Installation**: `pip install xgboost lightgbm catboost`

### Optuna
- **Version**: 3.0.0+
- **Use**: Hyperparameter optimization
- **Features**: Parallel trials, pruning, visualization
- **Installation**: `pip install optuna`

### AutoML Tools
- **Auto-sklearn**: Automated algorithm selection
- **H2O AutoML**: Comprehensive autoML
- **TPOT**: Tree-based pipeline optimization
- **Installation**: `pip install auto-sklearn h2o tpot`

---

## Data Processing & Analysis

### Pandas
- **Version**: 2.0.0+
- **Use**: Data manipulation, analysis
- **Features**: DataFrames, time series, IO operations
- **Installation**: `pip install pandas`

### Polars
- **Version**: 0.18.0+
- **Use**: Fast data processing
- **Strengths**: Rust backend, parallel processing, memory efficiency
- **Installation**: `pip install polars`

### NumPy / SciPy
- **Version**: 1.24.0+ / 1.10.0+
- **Use**: Numerical computing, scientific computing
- **Features**: Arrays, linear algebra, optimization
- **Installation**: `pip install numpy scipy`

### Apache Spark
- **Version**: 3.4.0+
- **Use**: Big data processing
- **Features**: Distributed computing, SQL, streaming
- **Installation**: Via PySpark - `pip install pyspark`

### Dask
- **Version**: 2023.8.0+
- **Use**: Parallel & distributed computing
- **Features**: Works with Pandas, NumPy, sklearn
- **Installation**: `pip install dask[complete]`

---

## Visualization & Dashboarding

### Matplotlib / Seaborn
- **Version**: 3.7.0+ / 0.12.0+
- **Use**: Static plots and visualizations
- **Installation**: `pip install matplotlib seaborn`

### Plotly / Plotly Dash
- **Version**: 5.15.0+ / 2.14.0+
- **Use**: Interactive visualizations, web dashboards
- **Features**: 3D plots, real-time updates, web-based
- **Installation**: `pip install plotly dash`

### Bokeh
- **Version**: 3.2.0+
- **Use**: Interactive web visualizations
- **Features**: Server support, streaming data
- **Installation**: `pip install bokeh`

### Streamlit
- **Version**: 1.26.0+
- **Use**: Rapid app development
- **Features**: Python-only, hot reload, deployment-ready
- **Installation**: `pip install streamlit`

---

## Natural Language Processing

### NLTK
- **Version**: 3.8.0+
- **Use**: NLP fundamentals
- **Features**: Tokenization, POS tagging, parsing
- **Installation**: `pip install nltk`

### spaCy
- **Version**: 3.6.0+
- **Use**: Production NLP
- **Models**: Trained pipelines for multiple languages
- **Installation**: `pip install spacy`

### Gensim
- **Version**: 4.3.0+
- **Use**: Topic modeling, word embeddings
- **Features**: Word2Vec, Doc2Vec, LDA
- **Installation**: `pip install gensim`

### TextBlob
- **Version**: 0.17.0+
- **Use**: Simplified NLP interface
- **Installation**: `pip install textblob`

---

## Computer Vision

### OpenCV
- **Version**: 4.8.0+
- **Use**: Image processing & computer vision
- **Features**: 3500+ algorithms
- **Installation**: `pip install opencv-python`

### Pillow (PIL)
- **Version**: 10.0.0+
- **Use**: Image manipulation
- **Features**: Resize, crop, filters
- **Installation**: `pip install pillow`

### scikit-image
- **Version**: 0.21.0+
- **Use**: Scientific image processing
- **Installation**: `pip install scikit-image`

### PyTorch Vision
- **Version**: 0.15.0+
- **Use**: Vision models & datasets
- **Features**: Pre-trained models, transforms
- **Installation**: `pip install torchvision`

---

## Time Series Analysis

### Statsmodels
- **Version**: 0.13.0+
- **Use**: Statistical modeling
- **Features**: ARIMA, exponential smoothing
- **Installation**: `pip install statsmodels`

### Prophet
- **Version**: 1.1.0+
- **Use**: Forecasting with trend & seasonality
- **Developer**: Meta (Facebook)
- **Installation**: `pip install pystan prophet`

### Sktime
- **Version**: 0.13.0+
- **Use**: Time series toolkit
- **Features**: Unified interface for multiple algorithms
- **Installation**: `pip install sktime`

### PyTorch Forecasting
- **Version**: 0.10.0+
- **Use**: Deep learning for time series
- **Installation**: `pip install pytorch-forecasting`

---

## Model Deployment & Serving

### FastAPI
- **Version**: 0.100.0+
- **Use**: High-performance APIs
- **Features**: Type hints, async support, auto-docs
- **Installation**: `pip install fastapi uvicorn`

### Flask
- **Version**: 2.3.0+
- **Use**: Lightweight web framework
- **Installation**: `pip install flask`

### Django
- **Version**: 4.2.0+
- **Use**: Full-featured web framework
- **Features**: ORM, admin panel, auth
- **Installation**: `pip install django`

### TensorFlow Serving
- **Use**: TF model serving
- **Installation**: Docker-based

### Seldon Core
- **Use**: Model serving on Kubernetes
- **Features**: Multi-model endpoints
- **Installation**: Kubernetes deployment

### BentoML
- **Version**: 1.0.0+
- **Use**: Model packaging & serving
- **Installation**: `pip install bentoml`

---

## MLOps & Experiment Tracking

### MLflow
- **Version**: 2.7.0+
- **Use**: ML lifecycle management
- **Features**: Tracking, registry, deployment
- **Installation**: `pip install mlflow`

### Weights & Biases (W&B)
- **Use**: Experiment tracking & visualization
- **Installation**: `pip install wandb`

### Neptune.ai
- **Use**: Model registry & collaboration
- **Installation**: `pip install neptune`

### DVC (Data Version Control)
- **Version**: 3.0.0+
- **Use**: Version control for data & models
- **Installation**: `pip install dvc`

### Kubeflow
- **Use**: ML workflows on Kubernetes
- **Features**: Training, serving, monitoring

---

## Containerization & Orchestration

### Docker
- **Version**: 24.0+
- **Use**: Containerization
- **Installation**: Download from docker.com

### Kubernetes
- **Version**: 1.27+
- **Use**: Container orchestration
- **Installation**: Via Docker Desktop or kubeadm

### Docker Compose
- **Version**: 2.0+
- **Use**: Multi-container applications
- **Installation**: Included with Docker Desktop

### Helm
- **Version**: 3.12.0+
- **Use**: Kubernetes package manager
- **Installation**: Download from helm.sh

---

## Infrastructure & Cloud

### Terraform
- **Version**: 1.5.0+
- **Use**: Infrastructure as Code
- **Installation**: Download from terraform.io

### AWS CLI / Boto3
- **Boto3 Version**: 1.28.0+
- **Use**: AWS programmatic access
- **Installation**: `pip install boto3 awscli`

### Google Cloud SDK
- **Use**: GCP access
- **Installation**: Download from cloud.google.com

### Azure SDK
- **Use**: Azure access
- **Installation**: `pip install azure-cli`

---

## Monitoring & Observability

### Prometheus
- **Version**: 2.45.0+
- **Use**: Metrics collection & alerting
- **Installation**: Docker image

### Grafana
- **Version**: 10.0.0+
- **Use**: Visualization & dashboards
- **Installation**: Docker image

### Elasticsearch, Logstash, Kibana (ELK)
- **Use**: Centralized logging
- **Installation**: Docker Compose stack

### Datadog
- **Use**: Comprehensive monitoring
- **Installation**: Agent installation

---

## Advanced Utilities

### SHAP (SHapley Additive exPlanations)
- **Version**: 0.42.0+
- **Use**: Model interpretability
- **Installation**: `pip install shap`

### LIME (Local Interpretable Model-Agnostic Explanations)
- **Version**: 0.2.0+
- **Use**: Local model explanation
- **Installation**: `pip install lime`

### Pydantic
- **Version**: 2.0.0+
- **Use**: Data validation
- **Installation**: `pip install pydantic`

### Requests
- **Version**: 2.31.0+
- **Use**: HTTP library
- **Installation**: `pip install requests`

### Logging & Debugging
- **Logging**: Built-in Python module
- **Rich**: Terminal formatting - `pip install rich`
- **Python Debugger**: Built-in pdb

---

## Installation Commands Summary

```bash
# Core ML Stack
pip install torch torchvision pytorch-lightning
pip install tensorflow keras
pip install scikit-learn xgboost lightgbm

# Data & Processing
pip install pandas numpy scipy polars dask[complete]

# NLP & Vision
pip install transformers nltk spacy opencv-python

# MLOps
pip install mlflow wandb dvc

# Web Frameworks
pip install fastapi uvicorn flask django

# Deployment
pip install docker kubernetes

# Visualization
pip install matplotlib seaborn plotly streamlit dash

# Others
pip install optuna shap lime pydantic requests
```

---

## Version Management

### Using Conda
```bash
# Create environment
conda create -n ml-env python=3.10

# Activate
conda activate ml-env

# Install packages
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch::pytorch pytorch::torchvision -c pytorch
```

### Using Poetry
```bash
poetry new my-project
poetry add pytorch torchvision
poetry install
```

### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

*Last Updated: 2025* | *All versions are as of publication date*
