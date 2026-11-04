| model | best_params | cv_acc | train_acc | test_acc | precision | recall | f1 | roc_auc | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| DT (library) | {"clf__max_depth": null, "clf__min_samples_leaf": 5} | 0.497 | 0.854 | 0.496 | 0.509 | 0.486 | 0.493 | 0.738 | 5.698 |
| DT + PCA (library) | {"clf__max_depth": 5, "clf__min_samples_leaf": 1, "pca__n_components": 0.8} | 0.456 | 0.532 | 0.447 | 0.359 | 0.387 | 0.369 | 0.730 | 15.941 |
| KNN (library) | {"clf__n_neighbors": 1, "clf__weights": "uniform"} | 0.517 | 0.999 | 0.520 | 0.521 | 0.521 | 0.512 | 0.712 | 0.010 |
| KNN + PCA (library) | {"clf__n_neighbors": 7, "clf__weights": "distance", "pca__n_components": 0.8} | 0.581 | 0.999 | 0.581 | 0.578 | 0.555 | 0.561 | 0.845 | 7.089 |
| DT (numpy) | {"clf__max_depth": 12, "clf__min_samples_leaf": 5} | 0.486 | 0.845 | 0.486 | 0.481 | 0.483 | 0.481 | 0.729 | 14.053 |
| DT + PCA (numpy) | {"clf__max_depth": 5, "clf__min_samples_leaf": 1, "pca__n_components": 0.8} | 0.456 | 0.532 | 0.447 | 0.359 | 0.387 | 0.369 | 0.730 | 3.132 |
| KNN (numpy) | {"clf__n_neighbors": 1, "clf__weights": "uniform"} | 0.517 | 0.999 | 0.524 | 0.526 | 0.524 | 0.516 | 0.714 | 0.001 |
| KNN + PCA (numpy) | {"clf__n_neighbors": 7, "clf__weights": "distance", "pca__n_components": 0.8} | 0.581 | 0.999 | 0.581 | 0.578 | 0.555 | 0.561 | 0.846 | 3.092 |
| ResNet-50 CNN (scratch) | SGD lr=0.02, 100 epochs | 0.798 | 0.818 | 0.802 | 0.777 | 0.799 | 0.780 | 0.965 | nan |

cv_acc = 5-fold CV accuracy on the train set (CNN: held-out validation accuracy); train_acc = accuracy on the training data (CNN: augmented batches at the best-val epoch).

- DT: library vs NumPy test predictions agree on 85.6%
- DT + PCA: library vs NumPy test predictions agree on 100.0%
- KNN: library vs NumPy test predictions agree on 99.6%
- KNN + PCA: library vs NumPy test predictions agree on 100.0%

Best model by macro ROC-AUC: **ResNet-50 CNN (scratch)** (AUC 0.965)
