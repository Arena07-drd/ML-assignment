| model | best_params | cv_acc | train_acc | test_acc | precision | recall | f1 | roc_auc | fit_s |
|---|---|---|---|---|---|---|---|---|---|
| DT (library) | {"clf__max_depth": 8, "clf__min_samples_leaf": 1} | 0.352 | 0.762 | 0.358 | 0.338 | 0.320 | 0.324 | 0.603 | 3.299 |
| DT + PCA (library) | {"clf__max_depth": 5, "clf__min_samples_leaf": 5, "pca__n_components": 0.8} | 0.396 | 0.470 | 0.372 | 0.311 | 0.320 | 0.306 | 0.671 | 4.306 |
| KNN (library) | {"clf__n_neighbors": 7, "clf__weights": "distance"} | 0.546 | 0.999 | 0.555 | 0.602 | 0.507 | 0.515 | 0.826 | 0.005 |
| KNN + PCA (library) | {"clf__n_neighbors": 3, "clf__weights": "distance", "pca__n_components": 0.8} | 0.557 | 0.999 | 0.579 | 0.589 | 0.539 | 0.552 | 0.804 | 14.109 |
| DT (numpy) | {"clf__max_depth": 8, "clf__min_samples_leaf": 3} | 0.359 | 0.736 | 0.354 | 0.321 | 0.313 | 0.313 | 0.606 | 14.714 |
| DT + PCA (numpy) | {"clf__max_depth": 5, "clf__min_samples_leaf": 1, "pca__n_components": 0.8} | 0.397 | 0.473 | 0.372 | 0.313 | 0.320 | 0.304 | 0.672 | 4.176 |
| KNN (numpy) | {"clf__n_neighbors": 7, "clf__weights": "distance"} | 0.546 | 0.999 | 0.555 | 0.602 | 0.507 | 0.515 | 0.826 | 0.004 |
| KNN + PCA (numpy) | {"clf__n_neighbors": 3, "clf__weights": "distance", "pca__n_components": 0.8} | 0.558 | 0.999 | 0.579 | 0.589 | 0.539 | 0.552 | 0.804 | 2.465 |
| ResNet-50 CNN (scratch) | SGD lr=0.02, 100 epochs | 0.798 | 0.818 | 0.802 | 0.777 | 0.799 | 0.780 | 0.965 | nan |

cv_acc = 5-fold CV accuracy on the train set (CNN: held-out validation accuracy); train_acc = accuracy on the training data (CNN: augmented batches at the best-val epoch).

- DT: library vs NumPy test predictions agree on 89.5%
- DT + PCA: library vs NumPy test predictions agree on 99.0%
- KNN: library vs NumPy test predictions agree on 100.0%
- KNN + PCA: library vs NumPy test predictions agree on 100.0%

Best model by macro ROC-AUC: **ResNet-50 CNN (scratch)** (AUC 0.965)
