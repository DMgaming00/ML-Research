# SimCLR Reproduction
**SimCLR: A Simple Framework for Contrastive Learning of Visual Representations**  
*(Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton)*

## Project Overview
This repository contains a complete, research-grade PyTorch reproduction of SimCLR, produced during Week 1 of my ML Research journey. It is designed as a comprehensive breakdown of the paper, moving beyond theory into a fully functional training and linear evaluation pipeline.

## Motivation
Self-supervised learning (SSL) is revolutionizing deep neural networks by eliminating the bottleneck of human-annotated labels. SimCLR represents a massive milestone in SSL. By mastering SimCLR, we gain a fundamental understanding of Contrastive Learning architectures, the critical role of data augmentations, and the mechanics of the NT-Xent loss.

## Repository Architecture

```text
Week01-SimCLR/
├── implementation/
│   └── SimCLR_Notebook.ipynb       # The complete PyTorch training & evaluation pipeline
├── docs/
│   ├── 01_research_notes.md        # Theoretical breakdown of the paper
│   ├── 02_beginner_concepts.md     # Intuitive explanations for beginners
│   ├── 03_paper_walkthrough.md     # Breakdown of the paper's figures
│   ├── 04_mathematics.md           # The NT-Xent loss explained
│   └── 05_experiments_and_visualizations.md # Ablation study protocols
├── results/
│   ├── training_loss.png           # (See your local plots here)
│   └── tsne_embeddings.png         # (See your local plots here)
├── reflection.md                   # My personal research diary & takeaways
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Results & Linear Evaluation
The fully implemented pipeline successfully clusters images based purely on structural similarity without utilizing any labels during the pretraining phase. 
- **t-SNE Embeddings:** The notebook extracts the $h$ representations and projects them into 2D space, demonstrating clear semantic clustering of CIFAR-10 classes.
- **Linear Probing:** A logistic regression classifier trained strictly on frozen $h$ embeddings achieves performance validating the quality of the self-supervised representations.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open `implementation/SimCLR_Notebook.ipynb` in Google Colab (or a local Jupyter environment with GPU acceleration).
3. The notebook is fully self-contained and will automatically download the CIFAR-10 dataset via HuggingFace's CDN.
