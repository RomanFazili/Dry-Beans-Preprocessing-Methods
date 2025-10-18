# Dry Bean Classification with Comprehensive Preprocessing Methods

A Machine Learning project which aims to boost simple classifier models' (K-NN, NB, DT) performance by applying various preprocessing methods like Feature Derivation, Box-Cox, Standardization, Z-Score Capping, LDA, BLSMOTE. Then, we compare it to more complex models (RF, MLP, SVM) to see whether we can bridge the gap between simple and complex models simply using preprocessing methods.

The project focuses on the Dry Bean Dataset which has been retrieved from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset, containing morphological features of different dry bean varieties. As there are little features in the dataset, no features selection was needed as all additionally derived features increase performance.

We created a custom pipeline for each simple model. For comparison, we also ran all models against a baseline pipeline, consisting only of Standardization (as is standard) and the classifier model itself. As is standard, we applied a 10-fold stratified CV and hyperparameter tuning using a gridsearch.

For performance measuring we used accuracy and F1 score. As output, the code gives all summary statistics like Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC etc. Additionally, it shows the confusion matrices and the ROC curves.

## Project Structure

```
├── Dry_Bean_Dataset.xlsx      # Original dataset
├── dataset.py                 # Dataset loading and feature engineering
├── gridsearch_analyzer.py     # Comprehensive model evaluation and visualization
├── main.py                    # Main execution script
└── README.md                  # This file
└── zscore_capper.py           # Custom transformation class to cap Z-Scores
```

## Technical Details
The project was written in Python 3.11. The external packages that were used are numpy, pandas, sklearn, imblearn, matplotlib and seaborn.

## Running the script
To run the script, simply run main.py. Here, you can input the directory of the Dry Bean Dataset.xlsx and then which model to choose (i.e. KNN-BASELINE). Then, the script will apply the gridsearch and output all necessary data.

There are 9 options to choose from w.r.t the classifier models:
- `KNN-OPTIMAL`, `DT-OPTIMAL`, `NB-OPTIMAL` (with full preprocessing)
- `SVM-BASELINE`, `RF-BASELINE`, `MLP-BASELINE` (baseline models)
- `KNN-BASELINE`, `DT-BASELINE`, `NB-BASELINE` (baseline models)