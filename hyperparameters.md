#  Hyper-parameter choices

In this project, we used transfer learning by extracting pretrained Inception‑V3 features (without end-to-end fine-tuning) and training a Random Forest classifier on top.

| Parameter                         | Value | Notes / Justification                                                              |
|----------------------------------:|------:|-----------------------------------------------------------------------------------:|
| Random seed                       | 235    | For reproducibility of results                                                     |
| CNN model                         | Inception‑V3 (pretrained on ImageNet) | Used as feature extractor; weights kept frozen      |
| Feature vector length             | 2048  | Output of Inception‑V3 pooling layer                                              |
| UMAP `n_neighbors`                | 15    | Default value; controls local/global balance in projection                        |
| UMAP `min_dist`                   | 0.1   | Controls tightness of clusters in UMAP                                            |
| Random Forest trees (`ntree`)     | 100   | Chosen to balance accuracy and computational time; empirically found stable        |
| learning rate                     | 0.0001 | Would be used if we fine-tuned the CNN                                            |
| batch size                        | 32    | Typical batch size for GPU-based training                                         |
| epochs                            | 40    | Typical small number, combined with early stopping to prevent overfitting         |
 
> epochs, batch sizes , learning rates are included here to document the typical defaults if we fine‑tune the CNN.
---
In addition to accuracy and ROC‑AUC, I would also like to add **F1 score**, which makes a summative measure on precision and recall — important in medical classification tasks with possible class imbalance.

