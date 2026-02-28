# Large Language Model From Scratch

This repository contains my implementation of a GPT-style Large Language Model built from scratch by following the book:

> **Build a Large Language Model (From Scratch)** by Sebastian Raschka

The project walks through building a transformer-based language model step-by-step, training it, and applying it to downstream tasks such as spam classification.

---

# Project Overview

This project covers:

- Tokenization  
- Positional embeddings  
- Multi-head self-attention  
- Transformer blocks  
- GPT-style decoder architecture  
- Language model training  
- Fine-tuning for classification  

---

# gpt2llm.ipynb – GPT Model Implementation

This notebook implements a GPT-style transformer model from scratch using PyTorch.

## Features Implemented

### ✔ Token Embeddings
Maps token IDs to dense vector representations.

### ✔ Positional Embeddings
Adds position information to token embeddings.

### ✔ Multi-Head Self Attention
- Scaled Dot-Product Attention  
- Causal masking  
- Multi-head parallel attention  

### ✔ Feed Forward Network
Two-layer MLP with non-linearity.

### ✔ Transformer Block
- Layer Normalization  
- Residual connections  
- Dropout  

### ✔ Full GPT Model
- Stack of Transformer blocks  
- Final layer normalization  
- Linear output projection  

---

# train_gpt.ipynb – Training the Language Model

This notebook handles:

- Dataset preparation  
- Dataloader creation  
- Loss computation (Cross Entropy)  
- Training loop  
- Validation loop  
- Model checkpointing  

## Training Details

- Optimizer: Adam / AdamW  
- Loss: Cross Entropy Loss  
- Teacher forcing training  
- Next-token prediction objective  

### Training Objective

The model learns:

```
P(token_t | token_1, token_2, ..., token_t-1)
```

This is standard autoregressive language modeling.

---

# spam_classificaltion.ipynb – Fine-Tuning GPT

This notebook demonstrates transfer learning by adapting the pretrained GPT model for spam detection.

## Steps Performed

1. Load pretrained GPT weights  
2. Replace output head with classification head  
3. Fine-tune on labeled spam dataset  
4. Evaluate accuracy  

## Key Concept

Instead of predicting next token, the model is trained to predict:

```
Spam (1) or Not Spam (0)
```

This shows how a language model can be repurposed for supervised downstream tasks.

---

# Architecture Summary

The implemented GPT architecture follows:

Input Tokens  
→ Token Embeddings  
→ Positional Embeddings  
→ N Transformer Blocks  
→ LayerNorm  
→ Linear Projection  
→ Softmax  

---

# Technologies Used

- Python  
- PyTorch  
- tiktoken (GPT-2 tokenizer)  
- Jupyter Notebook  

---

# Key Learnings

Through this implementation, I understood:

- How self-attention works mathematically  
- Why scaling factor √d_k is used  
- How causal masking prevents information leakage  
- Why LayerNorm stabilizes training  
- How residual connections enable deep networks  
- How autoregressive training works  
- How to fine-tune LLMs for classification tasks  

---

# Future Improvements

- Add learning rate scheduler  
- Add mixed precision training  
- Add text generation sampling (top-k, nucleus sampling)  
- Convert notebooks to modular Python scripts  
- Add model size scaling experiments  

---

# Reference

Sebastian Raschka  
*Build a Large Language Model (From Scratch)*  

---

# Author

Nishant Ray  
B.Tech Student | Machine Learning Enthusiast  
