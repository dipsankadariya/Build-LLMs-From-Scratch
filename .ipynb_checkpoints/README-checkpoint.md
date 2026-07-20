# Large Language Model (From Scratch) in PyTorch

This repository contains the complete code implementation of building llms from scratch where i built a complete **124-million-parameter GPT-2 (GPT-2 small)** from scratch using pure PyTorch, without relying on high-level transformer frameworks
The codebase follows the step-by-step implementation guide from the book **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka.

The goal of this project was to understand the "nuts and bolts" of how LLMs work by building everything from the ground up—from tokenization and attention mechanisms to pretraining, instruction fine-tuning, and building a ChatGPT-like assistant.

Each folder in this repository corresponds to different parts of the implementation, containing the main code files for the respective chapters.

Through this journey, I implemented every major component of a modern decoder-only transformer using only PyTorch, gaining a deep understanding of how large language models process text, learn from data, and generate coherent responses. This repository serves as both a learning resource and a complete reference implementation of an LLM built from first principles.

## 🚀 Topics Implemented from Scratch

The following core components and concepts are built entirely from scratch in this repository:

1. Tokenization (BPE)
2. Dataset Preparation
3. Sliding Window Data Generation
4. DataLoader
5. Token Embeddings
6. Positional Embeddings
7. Self-Attention (without trainable weights)
8. Self-Attention (with trainable weights)
9. Scaled Dot-Product Attention
10. Causal (Masked) Self-Attention
11. Dropout
12. Multi-Head Attention
13. Layer Normalization
14. GELU Activation
15. Feed-Forward Network (MLP)
16. Residual (Shortcut) Connections
17. Transformer Block
18. Stacking Transformer Blocks
19. GPT Model Architecture
20. Output Projection (Language Modeling Head)
21. Text Generation (Greedy Decoding)
22. Temperature Sampling
23. Top-k Sampling
24. Cross-Entropy Loss
25. Training Loop
26. Validation Loop
27. Optimizer (AdamW)
28. Learning Rate Warmup
29. Cosine Learning Rate Scheduling
30. Gradient Clipping
31. Model Evaluation
32. Text Generation During Training
33. Saving and Loading Model Checkpoints
34. Loading GPT-2 Pretrained Weights
35. Weight Transfer from GPT-2
36. Instruction Fine-Tuning
37. Dataset Formatting for Instruction Tuning
38. Supervised Fine-Tuning (SFT)
39. Chat-Style Prompt Formatting
40. LoRA (Low-Rank Adaptation)
41. Model Inference
42. Building a ChatGPT-like Assistant

## 📚 Chapters, Sections, and Subsections

The implementation closely follows the chapters of the book:

### Chapter 1 — Understanding Large Language Models
* 1.1 What is an LLM?
* 1.2 Applications of LLMs
* 1.3 Stages of Building and Using LLMs
* 1.4 Using LLMs Responsibly
* 1.5 Summary

### Chapter 2 — Working with Text Data
* 2.1 Understanding Word Embeddings
* 2.2 Tokenizing Text
  * 2.2.1 Byte Pair Encoding (BPE)
* 2.3 Converting Tokens into Token IDs
* 2.4 Adding Special Context Tokens
* 2.5 Byte Pair Encoding with tiktoken
* 2.6 Data Sampling with a Sliding Window
* 2.7 Creating Token Embeddings
* 2.8 Encoding Word Positions
* 2.9 Summary

### Chapter 3 — Coding Attention Mechanisms
* 3.1 The Problem with Modeling Long Sequences
* 3.2 Capturing Data Dependencies with Attention
* 3.3 Attending to Different Parts of the Input
  * 3.3.1 A Simple Self-Attention Mechanism Without Trainable Weights
  * 3.3.2 Computing Attention Weights for All Input Tokens
* 3.4 Implementing Self-Attention with Trainable Weights
* 3.5 Hiding Future Words with Causal Attention
* 3.6 Extending Single-Head Attention to Multi-Head Attention
  * 3.6.1 Stacking Multiple Single-Head Attention Layers
  * 3.6.2 Implementing Multi-Head Attention with Weight Splits
* 3.7 Summary

### Chapter 4 — Implementing a GPT Model from Scratch
* 4.1 Coding an LLM Architecture
* 4.2 Normalizing Activations with Layer Normalization
* 4.3 Implementing a Feed-Forward Network with GELU Activations
* 4.4 Adding Shortcut (Residual) Connections
* 4.5 Connecting Attention and Linear Layers in a Transformer Block
* 4.6 Coding the GPT Model
* 4.7 Generating Text
* 4.8 Summary

### Chapter 5 — Pretraining on Unlabeled Data
* 5.1 Evaluating Generative Text Models
* 5.2 Calculating the Text Generation Loss
* 5.3 Calculating the Training and Validation Loss
* 5.4 Training an LLM
* 5.5 Loading and Saving Model Weights
* 5.6 Loading Pretrained Weights from OpenAI
* 5.7 Generating Text with the Pretrained Model
* 5.8 Summary

### Chapter 6 — Fine-Tuning for Classification
* 6.1 Different Categories of Fine-Tuning
* 6.2 Preparing the Dataset
* 6.3 Creating Data Loaders
* 6.4 Initializing a Model with Pretrained Weights
* 6.5 Adding a Classification Head
* 6.6 Calculating Classification Loss and Accuracy
* 6.7 Fine-Tuning the Model on Labeled Data
* 6.8 Using the Fine-Tuned Model
* 6.9 Summary

### Chapter 7 — Fine-Tuning to Follow Instructions
* 7.1 Introduction to Instruction Fine-Tuning
* 7.2 Preparing an Instruction Dataset
* 7.3 Organizing Data into Training Batches
* 7.4 Loading a Pretrained LLM
* 7.5 Fine-Tuning the LLM on Instructions
* 7.6 Extracting and Saving LoRA Weights *(later editions include parameter-efficient fine-tuning)*
* 7.7 Evaluating the Fine-Tuned LLM
* 7.8 Summary

## 💡 Acknowledgements
Code and structure based on the book **Build a Large Language Model (From Scratch)** by Sebastian Raschka.

