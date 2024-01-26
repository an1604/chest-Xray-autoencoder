# Pneumonia Detection using Autoencoder and KNN

## Introduction
This project aims to detect pneumonia from chest X-ray images using a combination of an autoencoder for feature extraction and a K-Nearest Neighbors (KNN) classifier for classification. The autoencoder generates reconstructed images, and the KNN classifier utilizes these images to make predictions.

## Setup
Ensure you have the required packages installed:
- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `keras`
- `PIL`
- `opencv-python`
- `seaborn`

You can install them using pip:

```bash
pip install -r requirements.txt

## Project Structure
- train_chestXRAY.csv: CSV file containing training data information.
- validation_chestXRAY.csv: CSV file containing validation data information.
- test_chestXRAY.csv: CSV file containing test data information.
- autoencoder_weights.h5: Pre-trained weights for the autoencoder.
- autoencoder_model.h5: Saved model of the autoencoder.
- KNN_classifier.pkl: Pickled KNN classifier file.

## Usage
1. Run the provided Python script to train the autoencoder and KNN classifier.
2. The autoencoder generates reconstructed images.
3. The KNN classifier is trained using features extracted from the autoencoder.
4. Test the trained models on the test dataset.

## Files Description
- autoencoder_knn_pneumonia_detection.py: Python script containing the implementation of the autoencoder, KNN classifier, and pneumonia detection.
- autoencoder_knn_pneumonia_detection.ipynb: Jupyter Notebook version of the Python script.

## Results
- Accuracy for Images of 32x32 Original Size: 75.13%
- Accuracy for Images of 10x10 Size: 74.91%
- Accuracy for Images of 32x32 Size After PCA: 75.70%
- Best Accuracy achieved after PCA with 60 components: 76.34%

