# Garbage-Image-Classification-Using-Deep-Learning
Image classification with CNN

## 📌 Project Overview
**RecycleVision** is an end-to-end computer vision application designed to automate waste sorting and promote sustainable recycling habits[cite: 1, 2]. By leveraging deep learning and transfer learning architectures, this model classifies everyday waste items into **6 distinct categories**, helping streamline municipal waste management and smart recycling initiatives[cite: 1, 4].

---

## 🗂️ Dataset & Classes
* **Dataset Source:** Kaggle (TrashNet / Garbage Classification dataset)[cite: 1, 4]
* **Total Images:** ~2,527 JPG images[cite: 1, 4]
* **Target Classes (6):**
  1. 📦 `cardboard`[cite: 4]
  2. 🍾 `glass`[cite: 4]
  3. 🥫 `metal`[cite: 4]
  4. 📄 `paper`[cite: 4]
  5. 🥤 `plastic`[cite: 4]
  6. 🗑️ `trash`[cite: 4]

---

## 🚀 Key Features & Approach
1. **Exploratory Data Analysis (EDA):** Inspected class distributions, sample grids, and image dimensions to handle data imbalances and ensure dataset quality[cite: 1, 4].
2. **Preprocessing & Data Augmentation:** Built a robust pipeline using `ImageDataGenerator` with rescaling ($[0, 1]$), image resizing ($224 \times 224$), and real-time augmentation (rotation, zoom, horizontal flips) restricted strictly to the training split[cite: 1, 4].
3. **Model Development & Transfer Learning:** 
   * Built a custom baseline Convolutional Neural Network (CNN) from scratch[cite: 1, 4].
   * Implemented and compared multiple pre-trained transfer learning architectures (**MobileNetV2**, **ResNet50**, and **EfficientNetB0**)[cite: 4].
   * **Selected Model:** **MobileNetV2** (achieving ~76.5% validation accuracy with balanced precision/recall)[cite: 1, 4].
4. **Model Evaluation:** Generated classification reports (Precision, Recall, F1-Score) and Confusion Matrices to analyze model strengths and misclassifications[cite: 1, 4].
5. **Interactive Web Application:** Built and deployed an intuitive UI using **Streamlit**, allowing users to upload garbage images and receive real-time class predictions with confidence scores.

---

## 📊 Model Performance Summary
* **Validation Accuracy:** ~76%[cite: 4]
* **Top-Performing Categories:** `cardboard` (84% F1-score) and `paper` (85% F1-score)[cite: 4].

---

## 🛠️ Tech Stack
* **Language:** Python[cite: 4]
* **Deep Learning Framework:** TensorFlow / Keras[cite: 3, 4]
* **Metrics & Evaluation:** Scikit-Learn, Seaborn, Matplotlib[cite: 4]
* **Frontend / Deployment:** Streamlit[cite: 1, 2, 4]

---


