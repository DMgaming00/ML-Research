# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FOZy5PntjE8gKH595O_E3x1ZKjaKLFY_?usp=sharing)

**Paper:** https://arxiv.org/abs/2002.05709

A research-oriented reproduction of **SimCLR** implemented in PyTorch as part of my Machine Learning Research Journey. This project goes beyond implementing the paper by documenting the intuition, engineering decisions, mathematical concepts, experimental workflow, and critical analysis behind SimCLR.

---

## Project Overview

This repository contains a complete, research-oriented reproduction of **SimCLR**, one of the most influential papers in self-supervised representation learning.

The notebook covers:

- Evolution of representation learning leading to SimCLR
- Complete PyTorch implementation
- NT-Xent loss from scratch
- Training pipeline
- Representation visualization using t-SNE
- Linear Evaluation Protocol
- Ablation studies
- Engineering decisions
- Research notes and critical review

---

## Motivation

Self-supervised learning is changing how deep learning models are trained by reducing dependence on labeled datasets.

SimCLR demonstrated that carefully designed data augmentations and contrastive learning can produce high-quality visual representations without human annotations.

The goal of this project is not only to reproduce the paper but also to understand **why** each design decision works.

---

## Repository Structure

```text
Week01-SimCLR/
├── implementation/
│   └── SimCLR_Notebook.ipynb
├── docs/
│   ├── 01_research_notes.md
│   ├── 02_beginner_concepts.md
│   ├── 03_paper_walkthrough.md
│   ├── 04_mathematics.md
│   └── 05_experiments_and_visualizations.md
├── results/
│   ├── training_loss.png
│   └── tsne_embeddings.png
├── reflection.md
├── requirements.txt
└── README.md
```

---

## Results

The implementation successfully learns meaningful visual representations using self-supervised contrastive learning.

Current experiments include:

- Training loss visualization
- t-SNE embedding visualization
- Linear Evaluation Protocol
- Research-driven ablation experiments

---

## How to Run

### Option 1 (Recommended)

Click the **Open in Colab** badge at the top of this page.

### Option 2

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ML-Research.git
cd ML-Research/Week01-SimCLR
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter

```bash
jupyter notebook
```

Open

```
implementation/SimCLR_Notebook.ipynb
```

---

## Research Questions

Throughout this reproduction I explored questions such as:

- Why do augmentations act as supervision?
- Why does SimCLR require large batch sizes?
- Why is the projection head discarded during downstream tasks?
- Why does contrastive learning produce transferable representations?

---

## Future Work

- Reproduce BYOL
- Compare SimCLR vs BYOL
- Study DINO and Vision Transformers
- Explore Sparse Autoencoders
- Investigate mechanistic interpretability

---

## Disclaimer

This repository is an educational research reproduction created to deepen my understanding of representation learning and self-supervised learning. It is **not** an official implementation from the paper authors.

---

## Acknowledgements

- Chen et al. (2020), *SimCLR: A Simple Framework for Contrastive Learning of Visual Representations*
- PyTorch
- Hugging Face
- Google Research
