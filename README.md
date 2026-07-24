# 🧠 Large Language Model (From Scratch) in PyTorch

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Transformer](https://img.shields.io/badge/Architecture-GPT--2-orange?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-124M%20Parameters-success?style=for-the-badge)

</p>

## 📖 Overview

This repository contains the complete code implementation of building LLMs from scratch where I build a complete Large Language model(~124M) and every major components of it from scratch using pure PyTorch, without relying on high-level transformer frameworks. The goal of this project was to understand the **"nuts and bolts"** of how LLMs work by building everything from the ground up—from tokenization and attention mechanisms to pretraining, instruction fine-tuning, and building a ChatGPT-like assistant.Through this journey, I implemented every major component of a modern decoder-only transformer using only PyTorch, gaining a deep understanding of how large language models process text, learn from data, and generate coherent responses. This repository serves as both a learning resource and a complete reference implementation of an LLM built from first principles.

The codebase follows the step-by-step implementation guide from the book **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka.

<p align="center">
  <img src="/book.png" width="230" alt="Build a Large Language Model (From Scratch)">
  <br>
  <sub><i>Build a Large Language Model (From Scratch) — Sebastian Raschka</i></sub>
</p>

Each folder in this repository corresponds to different parts of the implementation, containing the main code files for the respective chapters.


## 🚀 Topics Implemented from Scratch

| Core Components | Training & Fine-Tuning |
|-----------------|------------------------|
| Tokenization (BPE) | Temperature Sampling |
| Dataset Preparation | Top-k Sampling |
| Sliding Window Data Generation | Cross-Entropy Loss |
| DataLoader | Training Loop |
| Token Embeddings | Validation Loop |
| Positional Embeddings | Optimizer (AdamW) |
| Self-Attention (without trainable weights) | Learning Rate Warmup |
| Self-Attention (with trainable weights) | Cosine Learning Rate Scheduling |
| Scaled Dot-Product Attention | Gradient Clipping |
| Causal (Masked) Self-Attention | Model Evaluation |
| Dropout | Text Generation During Training |
| Multi-Head Attention | Saving and Loading Model Checkpoints |
| Layer Normalization | Loading GPT-2 Pretrained Weights |
| GELU Activation | Weight Transfer from GPT-2 |
| Feed-Forward Network (MLP) | Instruction Fine-Tuning |
| Residual (Shortcut) Connections | Dataset Formatting for Instruction Tuning |
| Transformer Block | Supervised Fine-Tuning (SFT) |
| Stacking Transformer Blocks | Chat-Style Prompt Formatting |
| GPT Model Architecture | LoRA (Low-Rank Adaptation) |
| Output Projection (Language Modeling Head) | Model Inference |
| Text Generation (Greedy Decoding) | Building a ChatGPT-like Assistant |

## 📚 Chapters, Sections, and Subsections

The implementation closely follows the chapters of the book.

### 📘 Chapter 1 — Understanding Large Language Models

- **1.1** What is an LLM?
- **1.2** Applications of LLMs
- **1.3** Stages of Building and Using LLMs
- **1.4** Using LLMs Responsibly
- **1.5** Summary

### 📘 Chapter 2 — Working with Text Data

- **2.1** Understanding Word Embeddings
- **2.2** Tokenizing Text
  - **2.2.1** Byte Pair Encoding (BPE)
- **2.3** Converting Tokens into Token IDs
- **2.4** Adding Special Context Tokens
- **2.5** Byte Pair Encoding with tiktoken
- **2.6** Data Sampling with a Sliding Window
- **2.7** Creating Token Embeddings
- **2.8** Encoding Word Positions
- **2.9** Summary

### 📘 Chapter 3 — Coding Attention Mechanisms

- **3.1** The Problem with Modeling Long Sequences
- **3.2** Capturing Data Dependencies with Attention
- **3.3** Attending to Different Parts of the Input
  - **3.3.1** A Simple Self-Attention Mechanism Without Trainable Weights
  - **3.3.2** Computing Attention Weights for All Input Tokens
- **3.4** Implementing Self-Attention with Trainable Weights
- **3.5** Hiding Future Words with Causal Attention
- **3.6** Extending Single-Head Attention to Multi-Head Attention
  - **3.6.1** Stacking Multiple Single-Head Attention Layers
  - **3.6.2** Implementing Multi-Head Attention with Weight Splits
- **3.7** Summary

### 📘 Chapter 4 — Implementing a GPT Model from Scratch

- **4.1** Coding an LLM Architecture
- **4.2** Normalizing Activations with Layer Normalization
- **4.3** Implementing a Feed-Forward Network with GELU Activations
- **4.4** Adding Shortcut (Residual) Connections
- **4.5** Connecting Attention and Linear Layers in a Transformer Block
- **4.6** Coding the GPT Model
- **4.7** Generating Text
- **4.8** Summary
### 📘 Chapter 5 — Pretraining on Unlabeled Data

- **5.1** Evaluating Generative Text Models
- **5.2** Calculating the Text Generation Loss
- **5.3** Calculating the Training and Validation Loss
- **5.4** Training an LLM
- **5.5** Loading and Saving Model Weights
- **5.6** Loading Pretrained Weights from OpenAI
- **5.7** Generating Text with the Pretrained Model
- **5.8** Summary

### 📘 Chapter 6 — Fine-Tuning for Classification

- **6.1** Different Categories of Fine-Tuning
- **6.2** Preparing the Dataset
- **6.3** Creating Data Loaders
- **6.4** Initializing a Model with Pretrained Weights
- **6.5** Adding a Classification Head
- **6.6** Calculating Classification Loss and Accuracy
- **6.7** Fine-Tuning the Model on Labeled Data
- **6.8** Using the Fine-Tuned Model
- **6.9** Summary

### 📘 Chapter 7 — Fine-Tuning to Follow Instructions

- **7.1** Introduction to Instruction Fine-Tuning
- **7.2** Preparing an Instruction Dataset
- **7.3** Organizing Data into Training Batches
- **7.4** Loading a Pretrained LLM
- **7.5** Fine-Tuning the LLM on Instructions
- **7.6** Extracting and Saving LoRA Weights *(later editions include parameter-efficient fine-tuning)*
- **7.7** Evaluating the Fine-Tuned LLM
- **7.8** Summary

---

## 🎯 What I Built

By the end of this project, I successfully built:

- ✅ A complete **124-million-parameter GPT-2 (GPT-2 Small)** architecture from scratch.
- ✅ The full decoder-only Transformer architecture using only PyTorch.
- ✅ A complete pretraining pipeline.
- ✅ Text generation using greedy decoding, temperature sampling, and top-k sampling.
- ✅ Loading and using pretrained GPT-2 weights.
- ✅ Fine-tuning for downstream tasks.
- ✅ Instruction fine-tuning for chat-style interactions.
- ✅ LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
- ✅ A ChatGPT-like assistant capable of generating conversational responses.

---

## 💡 Acknowledgements

Special thanks to **Sebastian Raschka** for writing an outstanding book that makes the inner workings of Large Language Models accessible through hands-on implementation.

---
