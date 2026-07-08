# Final Deliverable Checklist

Before moving on to Paper #2, ensure every single item on this list is completed, pushed to GitHub, and rigorously understood.

## 1. Code Implementation
- [ ] `simclr.py` is fully modularized (Encoder, Projection Head, NT-Xent Loss).
- [ ] Data augmentations strictly follow the paper (Random Crop + Color Jitter composition).
- [ ] NT-Xent loss correctly masks the diagonal (ignoring self-similarity).
- [ ] Training loop runs without memory leaks.
- [ ] Linear Evaluation script is written (freezing the backbone and training a classifier on $h$).

## 2. Experiments & Metrics
- [ ] Loss curve drops and stabilizes (does not immediately hit zero).
- [ ] Ablation: Trained a model *without* color jitter (shortcut problem proved).
- [ ] Ablation: Trained a model with varying batch sizes (32 vs 128 vs 512).
- [ ] Ablation: Evaluated linear accuracy using $z$ vs using $h$.

## 3. Visualizations
- [ ] Generated t-SNE plot of embeddings showing semantic clustering on the test set.
- [ ] Plotted Cosine Similarity Heatmap of a batch (bright diagonal, dark off-diagonal).
- [ ] Built a Nearest Neighbor retrieval function and visualized top-5 matches for a query image.

## 4. Documentation & Career
- [ ] README is polished and professional.
- [ ] Research Notes are finalized and include questions raised during reading.
- [ ] Mathematical Breakdown is thoroughly understood (you can derive NT-Xent on a whiteboard).
- [ ] LinkedIn post is published.
- [ ] Can confidently answer all 20 interview questions without looking at the notes.

***Once this checklist is complete, you are officially ready for Paper #2 (MoCo or BYOL).***
