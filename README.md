# Pneumonia Detection using Autoencoder and KNN

## Introduction
Pneumonia Detection using Autoencoder and KNN is a project designed to automate the detection of pneumonia from chest X-ray images. By leveraging the capabilities of machine learning, particularly autoencoders for feature extraction and K-Nearest Neighbors (KNN) for classification, this project offers a robust solution for medical diagnostics.

Pneumonia is a common and potentially life-threatening condition that requires prompt diagnosis and treatment. Traditional methods of diagnosing pneumonia from chest X-ray images often rely on manual interpretation by trained radiologists, which can be time-consuming and subject to human error.

This project addresses these challenges by automating the detection process. The autoencoder is trained to extract meaningful features from chest X-ray images, which are then used to train a KNN classifier. The classifier is capable of accurately identifying patterns associated with pneumonia, enabling efficient and reliable diagnosis.

By combining advanced machine learning techniques with medical imaging, this project demonstrates the potential for automation in healthcare diagnostics. The automated detection of pneumonia can help streamline the diagnostic process, reduce the burden on healthcare professionals, and improve patient outcomes.

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
```
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

