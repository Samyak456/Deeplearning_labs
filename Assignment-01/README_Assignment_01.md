# Assignment 01 --- TensorFlow/Keras Setup and Fashion-MNIST Classification

## Overview

This practical demonstrates the basic deep-learning workflow in Google
Colab using TensorFlow/Keras. The assignment covers:

-   TensorFlow/Keras setup
-   Loading the Fashion-MNIST dataset
-   Train-test data handling
-   Pixel-value normalization
-   Dataset inspection and visualization
-   Building a Sequential neural network
-   Compiling and training the model
-   Evaluating test accuracy
-   Making predictions
-   Saving the trained Keras model
-   Inspecting the model architecture

**Student:** Samyak Shende\
**Roll No.:** 64\
**Subject:** Deep Learning\
**Practical:** 01\
**Academic Year:** 2026--27

## Problem Statement

> Install and configure TensorFlow/Keras in Google Colab. Perform data
> preprocessing, normalization, train-test split, and visualization on a
> sample dataset.

## Dataset

The notebook uses **Fashion-MNIST**, loaded directly through the Keras
dataset API:

``` python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST contains **60,000 training images and 10,000 test
images**. Each image is a **28×28 grayscale image** and belongs to one
of **10 fashion categories**.

### Official dataset documentation

https://keras.io/api/datasets/fashion_mnist/

### Class labels

    Label Class
  ------- -------------
        0 T-shirt/top
        1 Trouser
        2 Pullover
        3 Dress
        4 Coat
        5 Sandal
        6 Shirt
        7 Sneaker
        8 Bag
        9 Ankle boot

## Requirements

The assignment is designed for **Google Colab**. The notebook uses:

-   Python
-   TensorFlow
-   Keras (`tf.keras`)
-   Matplotlib

No manual dataset download is required because
`tf.keras.datasets.fashion_mnist.load_data()` downloads/caches the
dataset automatically.

## How to Run in Google Colab

### Step 1 --- Open Google Colab

Open:

https://colab.research.google.com/

Create a new notebook or upload `Assignment-01.ipynb`.

### Step 2 --- Runtime

For this introductory assignment, the default Colab runtime is
sufficient. A GPU is not required for the small MLP used here.

### Step 3 --- Run the notebook

Run the cells from top to bottom.

The notebook follows this sequence:

1.  Import TensorFlow and Matplotlib.
2.  Load Fashion-MNIST.
3.  Use the provided training/test split.
4.  Normalize image pixels from `[0, 255]` to `[0, 1]`.
5.  Inspect dataset shapes and labels.
6.  Visualize a sample image.
7.  Define the ten class names.
8.  Build the Sequential MLP.
9.  Compile the model.
10. Train for 5 epochs.
11. Evaluate on the test set.
12. Generate predictions.
13. Compare predicted and actual labels.
14. Save the model as `fashion_mnist_model.keras`.
15. Display the model summary.

## Methodology

### 1. Data loading

The Keras dataset API provides the training and testing partitions:

``` python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
```

### 2. Normalization

Pixel values are divided by 255:

``` python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

This changes the pixel range from approximately `0–255` to `0–1`, making
the numerical input scale more suitable for neural-network optimization.

### 3. Visualization

A training image is displayed with Matplotlib:

``` python
plt.imshow(train_images[0], cmap='gray')
plt.colorbar()
plt.show()
```

The notebook also maps the numeric label to its class name.

### 4. Model architecture

The implemented model is:

``` text
28×28 image
     ↓
Flatten
     ↓
784 values
     ↓
Dense(128, ReLU)
     ↓
Dense(10, Softmax)
     ↓
10-class prediction
```

The `Flatten` layer converts each 28×28 image into a vector of 784
values.

The hidden Dense layer has 128 neurons and uses ReLU.

The output layer has 10 neurons and uses Softmax to produce class
probabilities.

### 5. Compilation

The model uses:

-   Optimizer: Adam
-   Loss: Sparse Categorical Crossentropy
-   Metric: Accuracy

### 6. Training

The supplied notebook trains for:

``` python
epochs=5
```

### 7. Evaluation

The trained model is evaluated on the test set:

``` python
loss, accuracy = model.evaluate(test_images, test_labels)
```

The supplied notebook recorded a test accuracy of approximately **0.8749
(87.49%)** in its saved output.

### 8. Prediction

The notebook predicts class probabilities and selects the class with the
highest probability:

``` python
predicted_class = predictions[0].argmax()
```

The saved example predicted **Ankle Boot**, and the actual label was
also **Ankle Boot**.

### 9. Model saving

The trained model is saved as:

``` text
fashion_mnist_model.keras
```

## Important Clarification About Train-Test Split

The practical statement asks for a train-test split. In this
implementation, the split is supplied by the **Fashion-MNIST Keras
dataset loader itself** as `(train_images, train_labels)` and
`(test_images, test_labels)`. The notebook does not perform a separate
`train_test_split()` call.

This should be described in the viva/report as: **"The dataset is loaded
with its predefined training and testing partitions."**

## Expected Outputs

The notebook displays:

-   Training image shape: `(60000, 28, 28)`
-   Testing image shape: `(10000, 28, 28)`
-   First 10 training labels
-   A sample Fashion-MNIST image
-   The corresponding class name
-   Training progress for 5 epochs
-   Test loss and accuracy
-   Prediction probabilities for the first test image
-   Predicted and actual class
-   Model architecture and parameter summary

The saved notebook output reports:

``` text
Training Images: (60000, 28, 28)
Testing Images: (10000, 28, 28)
```

and:

``` text
Test Accuracy: 0.8748999834060669
```

## Project Files

Recommended repository structure:

``` text
Assignment-01/
├── Assignment-01.ipynb
├── README.md
├── Assignment_01_Fashion_MNIST_Instructions.pdf
└── fashion_mnist_model.keras   # optional; generated after running the notebook
```

## Full Notebook Code

The complete executable implementation is contained in
`Assignment-01.ipynb`. The main code blocks are:

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", accuracy)

predictions = model.predict(test_images)
predicted_class = predictions[0].argmax()

print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[test_labels[0]])

model.save("fashion_mnist_model.keras")
model.summary()
```

## Conclusion

This practical establishes the basic TensorFlow/Keras workflow for image
classification: loading data, preprocessing and normalization,
visualization, model construction, training, evaluation, prediction, and
model saving. The supplied notebook successfully demonstrates the
workflow on Fashion-MNIST and records approximately **87.49% test
accuracy** for the implemented 128-neuron MLP after five epochs.

## References

1.  Keras --- Fashion-MNIST dataset documentation:
    https://keras.io/api/datasets/fashion_mnist/
2.  Keras --- Getting Started: https://keras.io/getting_started/
3.  Keras --- Training & evaluation with built-in methods:
    https://keras.io/guides/training_with_built_in_methods/
