
# Image Data Driven Lung Cancer Classification Using Deep Learning and Optimization ALgorithm

## Project Overview
This project aims to classify lung cancer images using a Convolutional Neural Network (CNN) architecture. To improve model performance, Bayesian Optimization is used for hyperparameter tuning.

### Model Architecture
- **Input Layer:** Preprocessed lung cancer images
- **Hidden Layers:** Convolutional, MaxPooling, Dropout, Flatten, Dense layers
- **Activation Functions:** ReLU, Softmax
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

### Bayesian Optimization
The following hyperparameters were optimized:
- Learning Rate
- Batch Size
- Number of Epochs

### Optimized Hyperparameters
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10

## Results

### Accuracy and Loss
- The training and validation accuracy and loss over epochs are visualized in the attached plots:
  - `accuracy_plot.png`
  - `loss_plot.png`

### Confusion Matrix
- The confusion matrix (`confusion_matrix.png`) illustrates the classification performance across different lung cancer classes.

### Classification Report
The model achieved high precision, recall, and F1-score, especially in major classes:
- **Precision:** 0.95 (macro avg)
- **Recall:** 0.93 (macro avg)
- **F1-score:** 0.94 (macro avg)

Detailed metrics are available in `classification_report.txt`.

## Conclusion
The CNN model with Bayesian Optimization has demonstrated strong performance in classifying lung cancer images. The project highlights the impact of hyperparameter tuning on model effectiveness.

## Files Included
- `accuracy_plot.png`
- `loss_plot.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `README.md`

