# Assignment 2 — Multi-Layer Perceptron (MLP) Classifier on Wine Dataset

## Aim
To implement and evaluate Multi-Layer Perceptron (MLP) classifiers on the Wine dataset and compare their performance under different train-test split scenarios.

## Problem Statement
Build MLP-based classifiers and analyze how model architecture and the amount of training data affect classification performance.

## Dataset
The experiment uses the **Wine dataset** provided by `scikit-learn`.

- Samples: 178
- Input features: 13
- Classes: 3
- Task: Multiclass classification

## Scenarios
| Scenario | Training | Testing |
|---|---:|---:|
| Scenario 1 | 70% | 30% |
| Scenario 2 | 90% | 10% |

Stratified splitting with `random_state=42` is used for reproducibility.

## Models
**MLP1:** `(100, 100)` hidden layers  
**MLP2:** `(50,)` hidden layer

Both use:
- ReLU hidden-layer activations
- Adam optimization
- Initial learning rate: `0.001`
- Maximum iterations: `2000`
- `random_state=42`

## Preprocessing
The features are standardized with `StandardScaler`. The scaler is fitted only on the training set and then applied to the test set to avoid data leakage.

## Evaluation Metrics
- Accuracy
- Weighted F1 Score
- Weighted Precision
- Weighted Recall
- 10-fold cross-validation F1 Score
- Confusion Matrix
- MLP loss curve

## How to Run

### Google Colab
1. Upload `mlp_classifier_on_wine_dataset.ipynb` to Google Colab.
2. Run all cells from top to bottom.
3. No dataset download is required because the Wine dataset is included in scikit-learn.

### Local Python / VS Code
```bash
pip install -r requirements.txt
jupyter notebook mlp_classifier_on_wine_dataset.ipynb
```

## Project Structure
```text
Assignment_2_Wine_MLP/
├── README.md
├── mlp_classifier_on_wine_dataset.ipynb
├── mlp_classifier_on_wine_dataset_original.ipynb
├── requirements.txt
├── report/
│   ├── Assignment_2_Wine_MLP_Report.docx
│   └── Assignment_2_Wine_MLP_Report.pdf
└── results/
    ├── scenario1_mlp1.png
    ├── scenario1_mlp2.png
    ├── scenario2_mlp1.png
    └── scenario2_mlp2.png
```

## Note on the Original Notebook
The submitted notebook was cleaned for reproducibility. In the original function, cross-validation referenced `X_train` and `y_train` from outside the function instead of the scenario-specific training data. The revised notebook passes the correct scenario data into cross-validation. Feature scaling was also added because MLP training is sensitive to feature magnitude.

## Conclusion
The experiment demonstrates that MLP architecture and train-test split can affect classification performance. On this small, well-separated Wine dataset, both architectures can perform strongly after appropriate standardization. Cross-validation is included to provide a more reliable estimate than a single test split alone.

## Technologies
Python, NumPy, Pandas, Matplotlib, Scikit-learn, Jupyter Notebook / Google Colab.
