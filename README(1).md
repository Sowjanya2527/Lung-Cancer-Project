
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
  - ![Screenshot (34)](https://github.com/user-attachments/assets/8afc88db-04bc-4b14-b76e-081e746d5934)
  - ![Screenshot (35)](https://github.com/user-attachments/assets/cf20e9fb-f6c7-422f-a510-73299d273c75)

  
### Confusion Matrix
- The confusion matrix illustrates the classification performance across different lung cancer classes.
- ![Screenshot (39)](https://github.com/user-attachments/assets/4a5b6978-e502-465b-826d-6fdd6b02533e)


### Classification Report
The model achieved high precision, recall, and F1-score, especially in major classes:
- **Precision:** 0.95 (macro avg)
- **Recall:** 0.93 (macro avg)
- **F1-score:** 0.94 (macro avg)

Detailed metrics are available in 
c![Screenshot (38)](https://github.com/user-attachments/assets/c1b639bb-27c3-4a2a-8c26-0685add99c53)

## Conclusion
The CNN model with Bayesian Optimization has demonstrated strong performance in classifying lung cancer images. The project highlights the impact of hyperparameter tuning on model effectiveness.



