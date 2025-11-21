# CIFAR10-CNN-Classifier

## Overview
CIFAR10-CNN-Classifier is a deep learning project designed to classify images from the CIFAR‑10 dataset into ten categories using a Convolutional Neural Network (CNN). The project integrates Exploratory Data Analysis (EDA), data preprocessing, augmentation, model building, training with callbacks, evaluation, and interpretation of learned features via feature maps.

## Dataset
The project uses the **CIFAR‑10 dataset**, which contains 60,000 32×32 colour images in 10 classes, with 6,000 images per class. This dataset was selected for its diversity and suitability for image classification tasks.

## Features
- **Image pixels (R, G, B)**: Raw pixel values of the images.
- **Class labels**: Ten categories including airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Target Variable
- **Class label**: The category to which each image belongs (0–9).

## Methods
- **Exploratory Data Analysis (EDA)**: Visualisation of sample images, class distribution, pixel statistics, and t-SNE projections.  
- **Data Preprocessing**: Normalisation of pixel values and one-hot encoding of labels.  
- **Data Augmentation**: Random rotations, shifts, flips, and zooms applied to training images.  
- **CNN Model Architecture**: Three convolutional blocks with Batch Normalization, ReLU activation, MaxPooling, Dropout, followed by fully connected layers.  
- **Training**: Adam optimizer with early stopping and learning rate reduction on plateau.  
- **Evaluation**: Accuracy, loss curves, confusion matrix, and classification report.  
- **Interpretation**: Visualisation of feature maps from convolutional layers to understand learned representations.

## How to Run
Clone the repository:  
```bash
git clone "https://github.com/YourUsername/CIFAR10-CNN-Classifier"
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter notebooks or Python scripts:

bash
Copy code
CIFAR10_CNN_Training.ipynb
CIFAR10_CNN_Evaluation.ipynb
Results
Achieved strong classification accuracy on CIFAR‑10 test set.

Confusion matrices highlight common misclassifications.

Feature map visualisations show which parts of images the CNN focuses on during prediction.

References
Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Available: https://www.cs.toronto.edu/~kriz/cifar.html. Accessed: 21 November 2025.

TensorFlow Developers. CIFAR10 dataset – Keras. Available: https://keras.io/api/datasets/cifar10/. Accessed: 21 November 2025.

van der Maaten, L., & Hinton, G. (2008). Visualising data using t‑SNE. Journal of Machine Learning Research, 9, 2579–2605.

Selvaraju, R. R. et al. (2017). Grad‑CAM: Visual explanations from deep networks via gradient‑based localisation. Available: https://arxiv.org/abs/1610.02391. Accessed: 21 November 2025.

Keras Documentation. ImageDataGenerator API. Available: https://keras.io/api/preprocessing/image/. Accessed: 21 November 2025.

scikit-learn. Model evaluation: classification metrics. Available: https://scikit-learn.org/stable/modules/model_evaluation.html. Accessed: 21 November 2025.

Seaborn. Statistical data visualisation library. Available: https://seaborn.pydata.org/. Accessed: 21 November 2025.
