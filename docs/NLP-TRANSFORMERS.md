# 🗣️ Natural Language Processing & Transformers

## NLP Fundamentals

### Text Preprocessing
- Tokenization
- Lemmatization & Stemming
- Stop word removal
- Normalization

### Text Representations
- **Bag of Words (BoW)**: Word frequency vectors
- **TF-IDF**: Weighted word importance
- **Word2Vec**: Word embeddings (Skip-gram, CBOW)
- **GloVe**: Global Vectors for Word Representation
- **FastText**: Subword embeddings

---

## Transformer Architecture

### Core Components

#### 1. Self-Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- Queries (Q), Keys (K), Values (V)
- Multi-head attention for diverse representations
- Scaled dot-product attention

#### 2. Positional Encoding
- Captures word order information
- Sinusoidal positional embeddings
- Relative position biases

#### 3. Feed-Forward Networks
- Two dense layers per position
- Same across sequence

#### 4. Layer Normalization
- Stabilizes training
- Applied before/after sublayers

---

## Pre-trained Language Models

### BERT Family
- **BERT**: Bidirectional Encoder Representations
- **RoBERTa**: Robustly Optimized BERT
- **ALBERT**: A Lite BERT
- **DistilBERT**: Distilled BERT (40% smaller, 60% faster)

### GPT Family
- **GPT-2, GPT-3, GPT-4**: Decoder-only models
- Autoregressive language generation
- Few-shot learning capabilities
- Fine-tuning for downstream tasks

### Encoder-Decoder Models
- **T5**: Text-to-Text Transfer Transformer
- **BART**: Bidirectional Autoregressive Transformers
- **mT5**: Multilingual T5
- Machine translation, summarization, Q&A

### Multilingual Models
- **mBERT**: Multilingual BERT
- **XLM-RoBERTa**: Cross-lingual RoBERTa
- **mT5**: Multilingual T5

---

## NLP Tasks

### Text Classification
- Sentiment analysis
- Topic classification
- Intent detection
- Approach: Fine-tune BERT/RoBERTa

### Named Entity Recognition (NER)
- Identify entities (Person, Location, Organization)
- Sequence labeling task
- Approach: Token classification with Transformers

### Question Answering
- SQuAD-style extractive QA
- Generative QA with T5/GPT
- Retrieval-augmented generation (RAG)

### Machine Translation
- Seq2Seq models
- Attention mechanism
- Models: Transformer, mBART, mT5

### Text Summarization
- Abstractive: Generate summary
- Extractive: Select important sentences
- Models: T5, BART, Pegasus

### Named Entity Linking
- Link mentions to knowledge base
- Disambiguation
- Applications: Information extraction

### Semantic Similarity
- Sentence-BERT (SBERT)
- Cosine similarity matching
- Applications: FAQ matching, clustering

---

## Advanced NLP Techniques

### Transfer Learning
- Pre-training on large corpus
- Fine-tuning on specific tasks
- Domain adaptation

### Few-Shot Learning
- Prompt engineering
- In-context learning (GPT-style)
- Meta-learning approaches

### Data Augmentation
- Back-translation
- Paraphrasing
- Synthetic data generation

### Model Compression
- Distillation
- Quantization
- Pruning

### Domain-Specific Fine-tuning
- MEDBERT for medical NLP
- FinBERT for financial NLP
- LegalBERT for legal documents

---

## Tools & Libraries

### PyTorch-based
- **Hugging Face Transformers**: Model hub & training utilities
- **PyTorch Lightning**: Simplified training loop
- **Fairseq**: Sequence modeling toolkit

### TensorFlow-based
- **Transformers (TF)**: Hugging Face for TensorFlow
- **Keras**: High-level API

### Utilities
- **spaCy**: NLP pipeline
- **NLTK**: Toolkit for NLP
- **TextBlob**: Simplified NLP interface

---

## Performance Benchmarks

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| GLUE | ALBERT | Accuracy | 89.4% |
| SQuAD | ALBERT | F1 | 93.0% |
| BLEU (Translation) | mBART | BLEU | 28.5 |

---

## 🚀 Best Practices

1. **Start with pre-trained models** - Fine-tune rather than train from scratch
2. **Use appropriate tokenizers** - Match model's tokenizer
3. **Handle long sequences** - Truncate or use sliding window
4. **Balance datasets** - Especially for classification
5. **Monitor training** - Use validation metrics
6. **Deploy efficiently** - Use distilled models for inference

---

*More implementations available in projects folder.*
