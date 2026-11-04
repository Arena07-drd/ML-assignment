Open SVM_Waste_Classification.ipynb beside dataset-original and run all cells.
If needed: %pip install -r requirements.txt, then restart the kernel.

SVM and SVM + PCA use raw 32 x 32 color images: 3072 individual pixel values.
No color histograms, RGB summary statistics, HOG or LBP are used.
Both models use the same stratified 80/20 split. Training CV selects model parameters.
The notebook uses CPU, no pipelines, and includes saved execution outputs.
svm_results contains metrics, plots, the split manifest and a color-pixel cache.
Delete color_pixels_v4.npz if you change read_pixels or the image size.
