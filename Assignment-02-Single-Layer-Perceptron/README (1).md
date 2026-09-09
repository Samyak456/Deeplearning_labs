# Assignment 2 — Single-Layer Perceptron

## Aim
Implement a Single-Layer Perceptron for binary classification and analyze the effect of weights, bias, and activation functions.

## Problem Statement
Implement a Single-Layer Perceptron and study how the learned weights, bias, and different activation functions influence binary classification.

## Implementation
This notebook implements the perceptron learning rule directly using **NumPy**:
- OR gate dataset
- Two input features
- Step activation function
- Learning rate = `0.1`
- Epochs = `10`
- Manual weight and bias updates

The notebook also compares **Step, Sigmoid, and ReLU** outputs for the trained linear combination.

## Dataset
The experiment uses the OR gate truth table:

| X1 | X2 | Target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

## Algorithm
1. Initialize weights and bias.
2. Calculate the linear output `z = X·W + b`.
3. Apply the step activation function.
4. Calculate the error `target - prediction`.
5. Update weights using `W = W + learning_rate × error × X`.
6. Update bias using `b = b + learning_rate × error`.
7. Repeat for the selected number of epochs.
8. Test all input combinations.
9. Analyze different weights/bias values and activation functions.

## Key Concepts
### Forward computation
The perceptron calculates:
`z = X·W + b`

and then:
`prediction = step(z)`

### Learning rule
`W_new = W_old + η × error × X`

`b_new = b_old + η × error`

where `η` is the learning rate.

## Activation Functions
- **Step:** produces a binary 0/1 output and is used by the perceptron for classification.
- **Sigmoid:** maps values to the range 0–1.
- **ReLU:** returns 0 for negative inputs and the input value for positive inputs.

## Result
With the OR-gate data, the trained perceptron correctly classifies all four training samples after learning.

## Important Source Note
The uploaded practical sheet states **AND gate** in its algorithm, while the uploaded notebook actually implements an **OR gate**. This GitHub version documents the code that is actually implemented: OR gate classification. If the instructor specifically requires AND, change the target vector to `[0, 0, 0, 1]`.

## How to Run
### Google Colab
1. Upload `single_layer_perceptron.ipynb` to Google Colab.
2. Run all cells in order.
3. Review the epoch-wise weights/bias, predictions, weight/bias analysis, and activation-function comparison.

### Local
```bash
pip install -r requirements.txt
jupyter notebook single_layer_perceptron.ipynb
```

## Folder Structure
```text
Assignment_2_Single_Layer_Perceptron/
├── README.md
├── single_layer_perceptron.ipynb
├── single_layer_perceptron_original.ipynb
├── requirements.txt
├── results/
└── report/
    └── Assignment_2_Single_Layer_Perceptron_Report.pdf
```

## Conclusion
The Single-Layer Perceptron successfully learns the OR-gate decision boundary using the perceptron learning rule. The experiment demonstrates the roles of weights, bias, and activation functions in binary classification.
