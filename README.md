# Breast-Cancer-Classification
This project focuses on building a Convolutional Neural Network (CNN) model, named CancerNet, to classify breast cancer histology images as benign or malignant. The primary goal is to leverage deep learning techniques to aid in early cancer detection, improving diagnostic accuracy and supporting healthcare advancements.

Key Features
Utilizes the IDC_regular dataset for training and evaluation.
Implements data augmentation techniques for better generalization.
A custom CNN architecture is designed with multiple convolutional, pooling, and dense layers for high performance.
Evaluation metrics include accuracy, precision, recall, F1-score, and a confusion matrix for detailed analysis.
Training and validation processes visualized through accuracy and loss graphs.
Steps in the Project
Data Preparation: Automated dataset extraction and preprocessing.
Model Building: Design and compile the CancerNet CNN architecture.
Training and Validation: Train the model on augmented data and validate its performance.
Evaluation: Generate metrics and visualize results with confusion matrices.
Deployment: The trained model is saved for potential deployment in healthcare diagnostics.
Results
Achieved high classification accuracy on the validation dataset.
Confusion matrix and classification report provide insights into the model's performance.
How to Use
Clone this repository.
Place the dataset zip file in the root directory.
Run the provided script to preprocess data, train the model, and evaluate results.
Requirements
Python 3.7+
TensorFlow, NumPy, Matplotlib, Seaborn, and Scikit-learn.
Ensure all dependencies are installed via requirements.txt.
Future Work
Extend the model to classify other types of cancer images.
Optimize the CNN architecture for faster training and better performance.
Implement deployment pipelines for real-world applications.
