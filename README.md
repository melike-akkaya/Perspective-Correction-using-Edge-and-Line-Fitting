# Perspective Correction using Edge and Line Fitting

This repository contains the implementation and report for the first programming assignment of BBM418 - Computer Vision Laboratory (Spring 2024-2025), instructed by Dr. Nazlı İkizler Cinbis.

## 📌 Assignment Overview

The objective of this assignment is to correct perspective distortions in document images using:

- **Hough Transform** for initial line detection
- **RANSAC** to refine the detected lines
- **Geometric Transformations** to dewarp the documents

We work with a dataset that includes distorted document images and their ground-truth versions. The goal is to detect quadrilateral contours and restore the document's frontal view.

## 🛠️ Implementation

- Custom implementation of **Hough Transform**
- Use of **RANSAC** to handle noise and improve line fitting
- Perspective correction using geometric transformations
- Performance evaluation using **SSIM (Structural Similarity Index)**

## 📂 Dataset

The dataset is adapted from [WarpDoc Dataset](https://sg-vilab.github.io/event/warpdoc/) and contains scanned or photographed documents with various types of distortions. 
Each subfolder under "distorted/" and "digital/" corresponds to a different distortion type. The corrected versions are compared against the digital images using **SSIM**.

🔗 **Download Link** (smaller version of the dataset):  
[Google Drive](https://drive.google.com/file/d/1aPfzmYxLazyj15_zgCD96ImYdN9IZCPw/view?usp=sharing)
