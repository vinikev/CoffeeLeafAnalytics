# Coffee Leaf Disease Classification Using CNNs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

This project focuses on classifying the health of coffee leaves using binary classification with Convolutional Neural Networks (CNNs). It compares the performance of three distinct pre-trained architectures: **MobileNetV2**, **ResNet50**, and **VGG19**.

The core objective is to determine which model generalizes best for this specific agricultural task, using a dataset of coffee leaf images labeled as either **Healthy** or **Diseased**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
    - [1. Data Augmentation](#1-data-augmentation)
    - [2. Data Splitting](#2-data-splitting)
    - [3. Model Customization](#3-model-customization)
    - [4. Training & Evaluation](#4-training--evaluation)
- [Results & Discussion](#results--discussion)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [License](#license)

## Project Overview

The goal of this project is to build a robust image classifier that can accurately distinguish between healthy and diseased coffee leaves. We leverage transfer learning by fine-tuning three popular CNN models, applying the same data preparation and customization process to each for a fair comparison. The final model is evaluated based on its ability to generalize to unseen test data.

## Dataset

The chosen dataset is the **RoCoLe (Robusta Coffee Leaf) dataset**, which can be found on Kaggle: [RoCoLe.Original](https://www.kaggle.com/datasets/diegopgonzlez/rocoleoriginal).

- **Total Images:** 1,560 images of coffee leaves.
- **Labels:** Binary classification: **Healthy** or **Diseased**. (The original dataset also specifies the disease type, but for this project, we only use the binary health status).
- **Structure:** The dataset consists of an image directory and an `.xlsx` file containing the labels.

## Methodology

### 1. Data Augmentation

Due to the limited size of the original dataset (1,560 images), data augmentation was crucial to prevent overfitting and improve model generalization. We applied random transformations to each image to create 10 new variations.

- **Transformations Applied:**
    - **Rotation:** Random rotation between -30 and 30 degrees.
    - **Brightness:** Random brightness adjustment between 70% and 130%.
- **Result:** The dataset was expanded to a total of **15,600 images**. New annotation files were automatically generated to match the augmented images with their correct labels.

### 2. Data Splitting

The augmented dataset was split into three subsets to ensure proper model training, validation, and unbiased final evaluation:

- **Training Set:** 80% (for model learning)
- **Validation Set:** 10% (for tuning hyperparameters and monitoring overfitting)
- **Test Set:** 10% (for final performance evaluation on unseen data)

### 3. Model Customization

All three pre-trained models (`MobileNetV2`, `ResNet50`, `VGG19`) were customized using the same strategy for consistency:

1.  **Freezing Layers:** The first 20 layers of each base model were frozen. This preserves the generic features (like edge and shape detection) learned from ImageNet, speeds up training, and reduces the risk of overfitting on our smaller dataset.
2.  **Adding Custom Classifier Head:** A new classification head was added on top of the base model, consisting of:
    - `GlobalAveragePooling2D()`
    - `Dropout(0.5)` for regularization.
    - A `Dense` layer with 64 neurons, `'relu'` activation, and L2 regularization (`kernel_regularizer=l2(0.01)`).
    - A final `Dense` layer with 1 neuron and `'sigmoid'` activation for binary classification.

### 4. Training & Evaluation

Each model was trained for 10 epochs using the Adam optimizer and binary cross-entropy loss. The training process was monitored using the validation set, and final performance was measured on the held-out test set.

## Results & Discussion

The table below summarizes the key performance metrics from the experiment:

| Model           | Training Accuracy | Test Accuracy | Test Loss | Observation                                                                  |
| :-------------- | :---------------- | :------------ | :-------- | :--------------------------------------------------------------------------- |
| **MobileNetV2** | **99.2%**         | 85.8%         | 1.627     | Significant overfitting. High validation loss indicates poor generalization. |
| **ResNet50**    | 98.0%             | 86.3%         | 0.441     | Also showed signs of overfitting, with unstable validation performance.      |
| **VGG19**       | 92.2%             | **92.6%**     | **0.214** | Excellent generalization. Consistent low loss and highest test accuracy.     |

**Analysis:**

- While **MobileNetV2** achieved the highest training accuracy, it severely **overfitted** to the training data, failing to perform well on the validation and test sets. Its high test loss confirms this.
- **ResNet50** exhibited similar overfitting problems, with a test accuracy lower than its training accuracy would suggest.
- **VGG19** proved to be the most robust model. Although its training accuracy was the lowest, it demonstrated superior **generalization**, achieving the best test accuracy and the lowest test loss. This indicates its predictions are not only more accurate but also more confident and closer to the true values, even when it makes a mistake.

## Key Insights

- **Higher Training Accuracy ≠ Better Model:** This project clearly demonstrates that a model's ability to memorize training data (leading to high training accuracy) does not guarantee good performance on new, unseen data. Generalization is the true measure of a successful model.
- **VGG19's Robustness:** For this specific coffee leaf classification task, VGG19's architecture, despite being older and not achieving the highest training accuracy, proved to be the most effective at generalizing from the augmented dataset.
- **Importance of Validation:** The validation and test results were critical in identifying overfitting in MobileNetV2 and ResNet50, guiding the final model selection towards the more reliable VGG19.

## Technologies Used

- **Core Language:** Python 3.8+
- **Deep Learning Framework:** TensorFlow / Keras
- **CNN Architectures:** MobileNetV2, VGG19, ResNet50
- **Data Manipulation:** Pandas, NumPy
- **Image Processing:** Pillow (PIL)
- **Visualization:** Matplotlib
- **Environment:** Google Colab / Jupyter Notebook
