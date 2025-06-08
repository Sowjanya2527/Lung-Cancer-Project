## Image Data Driven Lung Cancer Classification Using Deep learning and Optimization Algorithm

### 📚 Project Overview
This project aims to classify lung cancer images into different stages (malignant, benign, or normal) using Convolutional Neural Networks (CNN) combined with Bayesian Optimization to fine-tune hyperparameters. The CNN model is trained on CT scan images, and Bayesian Optimization is used to optimize the hyperparameters (learning rate, batch size, and epochs) for improved accuracy and reduced training time.The purpose of this project is to assist healthcare professionals by providing an early-stage diagnostic tool for lung cancer, ultimately leading to better treatment outcomes for patients The dataset used in this project is from the IQ-OTHNCCD lung cancer dataset, containing labeled images of lung CT scans

####⚙️ Technologies Used
    Python: Programming language
    TensorFlow / Keras: Deep learning framework
    Convolutional Neural Networks (CNN): Neural network architecture for image classification
    Bayesian Optimization: Optimization technique for hyperparameter tuning
    OpenCV: Image processing
    NumPy & Pandas: Data manipulation
    Matplotlib / Seaborn: Data visualization
    Scikit-learn: For metrics evaluation
    
 Key Features
    Convolutional Neural Network (CNN) for image classification
    Bayesian Optimization for hyperparameter tuning (learning rate, batch size, epochs)
    Classification of 3 categories: Benign, Malignant, Normal
    High accuracy achieved with an optimized model
    Prediction with confidence levels and similarity checks to ensure valid CT scan images

Table of Contents
    Project Description
    Installation
    Usage
    Results
    Model Evaluation
    References

Model Architecture
        CNN (Convolutional Neural Network):
        Two convolutional layers for feature extraction
        Max-pooling for down-sampling
        A dense fully connected layer for classification
        Dropout layer to prevent overfitting
        Softmax output layer for multi-class classification

Optimization
    Bayesian Optimization is used to search for the best hyperparameters (learning rate, batch size, epochs) by evaluating different combinations and selecting the one that yields the best validation accuracy.

Installation

Prerequisites
    Python 3.x
    TensorFlow 2.x
    Keras Tuner
    OpenCV
    Matplotlib
    scikit-learn

Setup Instructions
    Clone the repository:

git clone https://github.com/yourusername/lung-cancer-classification.git
cd lung-cancer-classification

Install the dependencies:
    pip install -r requirements.txt
    Prepare the dataset:
        Download the IQ-OTHNCCD lung cancer dataset and place it in the dataset_path directory as described in the script.

Usage
1. Load Images and Preprocess Data
The images from the dataset are loaded, resized, and normalized (scaled between 0 and 1).

2. Hyperparameter Tuning
Bayesian Optimization is used to tune the hyperparameters (learning rate, batch size, and epochs) for the CNN model.

3. Train and Evaluate the Model
The final CNN model is trained with the best hyperparameters, and its performance is evaluated using test data.

4. Predict New Images
The model can be used to predict whether a given lung CT scan is Benign, Malignant, or Normal. It will display the predicted class and confidence level.

Example to predict a lung CT scan image:

python predict.py --image_path "path/to/image.jpg"

Results
Final Model Accuracy
    Training Accuracy: 99.09%
    Validation Accuracy: 99.55%
    Test Accuracy: 99.55%

Classification Report
              precision    recall  f1-score   support
      Benign       0.96      1.00      0.98        24
   Malignant       1.00      0.99      1.00       113
      Normal       0.99      0.99      0.99        83
    accuracy                           0.99       220
   macro avg       0.98      0.99      0.99       220
weighted avg       0.99      0.99      0.99       220

Confusion Matrix

A confusion matrix is plotted to show the model's performance in predicting the classes.

Model Evaluation
    Bayesian Optimization helped find the best hyperparameters for the CNN model.
    The model achieved 99.55% accuracy on both the validation and test datasets.
    The confusion matrix and classification report indicate excellent performance with high precision, recall, and F1-score across all classes.

Sample Prediction

When a CT scan image is provided as input, the model predicts whether the scan is:
    Benign: Indicates a non-cancerous condition.
    Malignant: Indicates a cancerous condition.
    Normal: Indicates a non-abnormal condition.

Example Output:
🔬 Predicted Class: Malignant
📊 Confidence: 0.97

References
    Bayesian Optimization:
        Keras Tuner Documentation
        Bayesian Optimization Paper
    Lung Cancer Dataset:
        IQ-OTHNCCD lung cancer dataset: Dataset Link
    Lung Cancer Classification Using CNN:
        Convolutional Neural Networks for Image Classification
