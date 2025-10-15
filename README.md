﻿# Statistical Learning & Pattern Recognition Projects

This repository contains a series of projects completed for UCSD's ECE 271A: Statistical Learning I course. The projects demonstrate the practical implementation of fundamental machine learning algorithms for tasks such as image segmentation, classification, and density estimation.

## Core Competencies Demonstrated

-   **Classification:** Bayesian Decision Theory, Gaussian Classifiers (ML, Bayesian), K-Nearest Neighbors
-   **Dimensionality Reduction:** Principal Component Analysis (PCA), Fisher's Linear Discriminant Analysis (LDA)
-   **Density Estimation:** Parametric (MLE, Bayesian Estimation) and Non-Parametric (Kernel Density) Methods
-   **Clustering & Latent Variables:** Gaussian Mixture Models (GMM), Expectation-Maximization (EM) Algorithm
-   **Model Evaluation:** Bias-Variance Tradeoff, Error Rate Calculation, Confusion Matrices

## Technology Stack

-   **Primary:** Python, NumPy, SciPy, Matplotlib, imageio, Matlab

-----

## Project Showcase

This table provides a high-level overview of each project. Click on the project name to jump to the detailed section.

| Project                                                              | Core Concepts Applied                             | Key Result Metric        |
| -------------------------------------------------------------------- | ------------------------------------------------- | ------------------------ |
| [**HW1: Bayesian Classifier for Image Segmentation**](#hw1-bayesian-classifier-for-image-segmentation) | Bayesian Decision Theory, DCT Feature Extraction  | **Error Rate: 17.27%** |
| **HW2: Gaussian Classifiers** *(Details to be added)* | Maximum Likelihood Estimation (MLE), Covariance Analysis | Classification Accuracy  |
| **HW3: Dimensionality Reduction** *(Details to be added)* | PCA, Fisher's LDA                                 | Visualization, Separation|
| **HW4: Non-Parametric Methods** *(Details to be added)* | K-Nearest Neighbors, Kernel Density Estimation    | Classification Accuracy  |
| **HW5: Mixture Models & EM** *(Details to be added)* | Gaussian Mixture Models, Expectation-Maximization | Log-Likelihood, Clustering |

-----

## Project Details

### HW1: Bayesian Classifier for Image Segmentation

-   **Objective:** To build a classifier that segments an image into "cheetah" (foreground) and "grass" (background) classes using Bayesian Decision Theory.

-   **Methodology:**
    1.  **Feature Extraction:** The image was processed with a sliding 8x8 window. For each block, the Discrete Cosine Transform (DCT) was computed. The feature was the index (1-64) of the DCT coefficient with the second-largest magnitude, capturing the block's dominant frequency component.
    2.  **Probability Estimation:** Class priors `P(Y)` were estimated from sample counts. Class-conditional likelihoods `P(X|Y)` were modeled as normalized histograms of the features, with Laplace smoothing applied for robustness.
    3.  **Classification:** The Bayes Decision Rule for minimum error was applied to classify the center pixel of each 8x8 block. A full-size segmentation mask was generated, and boundary pixels were filled using nearest-neighbor interpolation.
    4.  **Evaluation:** The final mask was compared against a ground truth to calculate the probability of error, a confusion matrix, precision, and recall.

-   **Result:**
    -   **Probability of Error:** **17.27%**
    -   **Performance Analysis Visualization:** The figure below shows the generated mask, the ground truth, and a color-coded map of classification errors (False Positives in red, False Negatives in blue).

      ![Generated Segmentation Mask vs Ground Truth](hw1/hw1_error_analysis.png)

-   **File Structure for HW1:**
    -   `hw1_solution.ipynb`: Jupyter Notebook with the complete implementation and analysis.
    -   `cheetah.bmp`: The input test image.
    -   `cheetah_mask.bmp`: The ground-truth mask for error calculation.
    -   `TrainingSamplesDCT_8.mat`: Training data containing DCT coefficients.
    -   `Zig-Zag Pattern.txt`: The scanline order for converting 8x8 DCT matrices to 64x1 vectors.