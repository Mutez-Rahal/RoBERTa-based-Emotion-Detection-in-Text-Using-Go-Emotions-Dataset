# RoBERTa-based Emotion Detection in Text Using Go-Emotions Dataset

This repository contains a multi-label emotion detection system for textual data, developed as part of a research project. The proposed approach leverages transformer-based language models to identify multiple emotions expressed in text, with a focus on robustness, interpretability, and practical applicability.

The system is evaluated on the **Go-Emotions** dataset and maps fine-grained emotion annotations to **Ekman’s five basic emotions**: *anger, fear, joy, sadness,* and *surprise*.

---

## Project Overview

Emotion detection is a core task in natural language processing with applications in mental health monitoring, customer feedback analysis, social media moderation, and human–computer interaction. Unlike traditional sentiment analysis, emotion detection must handle overlapping emotions, subtle contextual cues, and imbalanced class distributions.

This project addresses these challenges through:
- Multi-label emotion classification
- Transformer-based contextual representations
- Ekman-level emotion mapping
- Class imbalance–aware training strategies

---

##  Key Features

- **Transformer-based Model**: Fine-tuned **RoBERTa-base** for emotion classification  
- **Multi-Label Framework**: Simultaneous prediction of multiple emotions using sigmoid outputs  
- **Ekman Emotion Mapping**: Practical reduction of 27 Go-Emotions labels into 5 fundamental emotions  
- **Class Imbalance Handling**: Weighted loss functions and adaptive thresholding  
- **Robust Preprocessing Pipeline**: Cleaning, filtering, and deduplication strategies  
- **Reproducible Training Setup**: Clear hyperparameters and evaluation protocol  

---

## Dataset

This project uses the **Go-Emotions** dataset:

- Source: Google Research
- Size: ~58k samples
- Original Labels: 27 fine-grained emotions + neutral
- Reference:  
  > Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020

### Emotion Mapping
The original labels are mapped to Ekman’s basic emotions to balance emotional granularity and classification reliability.

| Ekman Emotion | Example Mapped Labels |
|--------------|-----------------------|
| Joy | joy, amusement, pride |
| Sadness | sadness, grief, disappointment |
| Anger | anger, annoyance, disapproval |
| Fear | fear, nervousness |
| Surprise | surprise, realization |

---

##  Methodology

### Model Architecture
- Pre-trained **RoBERTa-base**
- `[CLS]` token representation used for sentence-level embedding
- Fully connected classification head
- Sigmoid activation for multi-label prediction

### Training Details
- Optimizer: AdamW  
- Learning Rate: 1e-5  
- Batch Size: 32  
- Epochs: 5  
- Loss Function: Weighted BCE  
- Scheduler: Linear warmup and decay  
- Early Stopping: Validation macro-F1  

---

## Experimental Results

Evaluation is performed on a held-out test set using standard multi-label metrics.

**Test Set Performance:**
- Micro-F1: **0.8245**
- Macro-F1: **0.7558**
- Precision: **0.7217**
- Recall: **0.7972**
- Hamming Loss: **0.0769**

Results demonstrate strong overall performance and consistent behavior across all emotion categories, including minority classes.

---

##  Getting Started

### Requirements
```bash
python >= 3.8
torch
transformers
scikit-learn
numpy
pandas
```

### Clone the Repository
```bash
git clone https://github.com/Mutez-Rahal/RoBERTa-based-Emotion-Detection-in-Text-Using-Go-Emotions-Dataset.git
cd emotion-detection-roberta


