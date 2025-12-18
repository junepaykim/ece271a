# Statistical Learning & Pattern Recognition Projects

This repository contains a series of projects completed for UCSD's ECE 271A: Statistical Learning I course. The projects demonstrate the practical implementation of fundamental machine learning algorithms for tasks such as image segmentation, classification, and density estimation.

## Core Competencies Demonstrated

-   **Classification:** Bayesian Decision Theory, Gaussian Classifiers (ML, Bayesian), K-Nearest Neighbors
-   **Parameter Estimation:** Maximum Likelihood (MLE), Maximum A Posteriori (MAP), Bayesian Predictive Densities
-   **Dimensionality Reduction:** Principal Component Analysis (PCA), Fisher's Linear Discriminant Analysis (LDA)
-   **Clustering & Latent Variables:** Gaussian Mixture Models (GMM), Expectation-Maximization (EM) Algorithm
-   **Model Evaluation:** Bias-Variance Tradeoff, Error Rate Calculation, Confusion Matrices

## Technology Stack

-   **Primary:** Python, NumPy, SciPy, Matplotlib, imageio, Matlab

-----

## Project Showcase

This table provides a high-level overview of each project. Click on the project name to jump to the detailed section.

| Project                                                              | Core Concepts Applied                             | Key Result Metric       |
| -------------------------------------------------------------------- | ------------------------------------------------- |-------------------------|
| [**HW1: Bayesian Classifier for Image Segmentation**](#hw1-bayesian-classifier-for-image-segmentation) | Bayesian Decision Theory, DCT Feature Extraction  | **Error Rate: 17.27%**  |
| [**HW2: Gaussian Classifiers for Segmentation**](#hw2-gaussian-classifiers-for-segmentation) | Multivariate Gaussians, MLE, Feature Selection (KL Divergence) | **Error Rate: 7.48%**   |
| [**HW3: Bayesian Parameter Estimation**](#hw3-bayesian-parameter-estimation) | MAP, Bayesian Predictive, Conjugate Priors        | **Error Rate: ~11.69%** |
| [**HW4: Mixture Models & EM**](#hw4-mixture-models--em)              | Gaussian Mixture Models, Expectation-Maximization | **Error Rate: ~6.50%**  |

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

### HW2: Gaussian Classifiers for Segmentation

-   **Objective:** To extend the Bayesian classifier by modeling class-conditional densities as Multivariate Gaussian distributions and to demonstrate the "Curse of Dimensionality" by comparing a 64-dimensional model against a reduced 8-dimensional model.

-   **Methodology:**
    1.  **Gaussian Modeling (MLE):** The class-conditional densities `P(X|Y)` were assumed to be multivariate Gaussian. The Mean vectors ($\mu$) and Covariance matrices ($\Sigma$) were estimated using Maximum Likelihood Estimation (MLE) on the training data.
    2.  **Feature Selection:** The Symmetric Kullback-Leibler (KL) Divergence was calculated for all 64 features to quantify their discriminative power. The top 8 features were selected to build a low-dimensional classifier.
    3.  **Classification:** A sliding window approach was used. Pixels were classified using the Bayesian Decision Rule, implemented via log-likelihood functions for numerical stability.
    4.  **Comparison:** Two classifiers were evaluated: one using the full 64-dimensional feature vector and one using only the best 8 features.

-   **Result:**
    -   **64D Error Rate:** 14.50% (High False Positives due to overfitting).
    -   **8D Error Rate:** **7.48%** (Best Performance).
    -   **Insight:** The 8D classifier was better than the 64D classifier. This confirms the Curse of Dimensionality: estimating a full $64 \times 64$ covariance matrix (2080 parameters) with limited data leads to poor generalization, whereas the simpler 8D model (36 parameters) is more robust.

      ![8D Gaussian Segmentation Result](hw2/output/segmentation_8d_KL.png)

-   **File Structure for HW2:**
    -   `hw2_solution.py`: Python script for MLE parameter estimation, KL divergence calculation, and image classification. 
    -   `output/`: Directory containing generated plots, including marginal densities and segmentation masks.
    -   `TrainingSamplesDCT_8_new.mat`: Updated training data for Gaussian estimation.
### HW3: Bayesian Parameter Estimation

-   **Objective:** To explore Bayesian Parameter Estimation by treating the class-conditional mean $\mu$ as a random variable rather than a fixed constant. This project compares the performance of Maximum Likelihood (ML), Maximum A Posteriori (MAP), and Bayesian Predictive estimators across varying dataset sizes ($N$) and prior uncertainties ($\alpha$).

-   **Methodology:**
    1.  **Model Setup:** Class-conditional densities were modeled as multivariate Gaussians with known covariance (approximated by the sample covariance of each training subset) but unknown mean.
    2.  **Prior Distribution:** A Gaussian prior $P(\mu) \sim \mathcal{N}(\mu_0, \Sigma_0)$ was applied to the mean, where $\Sigma_0 = \alpha\,\mathrm{diag}(W_0)$.
    3.  **Estimator Comparison:**
        * **ML:** Uses the training-set sample mean for each class.
        * **MAP:** Uses the posterior mean $\mu_n$ as a plug-in estimate.
        * **Bayesian Predictive:** Uses the predictive covariance $\Sigma + \Sigma_n$ to account for posterior uncertainty in $\mu$.
    4.  **Decision Rule vs. Evaluation Convention:**
        * **Class priors in the decision rule:** ML-estimated from the training subset (dataset-dependent).
        * **Probability of Error (PoE):** Computed on the *cheetah test image* as the **pixel-wise misclassification rate** using the ground-truth mask.
    5.  **Strategies:** Tested two prior strategies provided by the starter files:
        * **Strategy 1:** Prior mean closer to the ML means (more informative).
        * **Strategy 2:** Prior mean farther from the ML means (less informative / poorer prior).

-   **Result:**
    -   **Convergence behavior:** As $\alpha$ becomes large, both MAP and Bayesian Predictive curves converge to the ML baseline (the prior becomes effectively uninformative).
    -   **Prior sensitivity:** For small $\alpha$, MAP/Predictive performance can improve or degrade relative to ML depending on how well the prior mean aligns with the test-image distribution and the dataset size.
    -   **Representative curve (Strategy 1, Dataset 1, $d=64$):**

      ![PoE vs Alpha Strategy 1 Dataset 1](hw3/output/PoE_Strategy_1_Dataset_1_d64.png)

-   **File Structure for HW3:**
    -   `hw3/hw3_solution.py`: Final implementation that generates the PoE-vs-alpha curves (saved with suffix `_d64.png`).
    -   `hw3/output/`: Contains plots for both strategies and all datasets:

### HW4: Mixture Models & EM Algorithm

- **Objective:** Implement a Gaussian Mixture Model (GMM) classifier trained with the Expectation–Maximization (EM) algorithm for cheetah (FG) vs grass (BG) segmentation, and study:
  1) sensitivity to random EM initialization (fixed \(C=8\)), and  
  2) effect of mixture complexity \(C\) on the probability of error (PoE).

- **Methodology:**
  1. **Model:** Class-conditional densities \(p(\mathbf{x}\mid Y)\) are modeled as GMMs with **diagonal covariance** matrices (one GMM for FG and one for BG).
  2. **Feature Extraction:** Sliding \(8\times8\) window DCT features (64-D) from `cheetah.bmp` using the provided zig-zag ordering.
  3. **Feature Ranking (B-order):** Instead of taking the first \(d\) zig-zag coefficients, we **rank features by separability** using \(\alpha\) computed from the provided subset data (from `TrainingSamplesDCT_subsets_8.mat`, using D4_FG/D4_BG), and evaluate using the **top-\(d\)** features by this ranking.
  4. **Bayes Decision Rule:** Classify using log-likelihood ratio with empirical priors from training counts (FG=250, BG=1053).
  5. **Experiments:**
     - **Part (a) Initialization Sensitivity:** Train **5** FG GMMs and **5** BG GMMs with \(C=8\) using different random initializations → **25** FG/BG pairs. For each pair, compute PoE over  
       \(d \in \{1,2,4,8,16,24,32,40,48,56,64\}\).
     - **Part (b) Varying Components:** Train one FG/BG pair for each \(C \in \{1,2,4,8,16\}\) and evaluate PoE over the same \(d\) list.

- **Key Results:**
  - **Empirical Priors:** FG ≈ **0.1919**, BG ≈ **0.8081**
  - **Part (a):** Initialization sensitivity is **small** overall. Example at \(d=32\):  
    mean PoE ≈ **0.0606**, range ≈ **[0.0566, 0.0642]**
  - **Best dimension region:** PoE typically minimized around **moderate \(d\)** (roughly **24–40**).  
    In this run, the minimum mean occurs near **\(d=32\)**.
  - **Part (b):** Mixture complexity matters:  
    - \(C=1\) degrades at larger \(d\) (PoE grows to ~0.10+)  
    - **Moderate \(C\)** (often \(C=2\) or \(C=4\)) achieves the **lowest** PoE (mid \(0.05\)–\(0.06\) range)  
    - Larger \(C\) (8, 16) does **not** consistently improve due to EM local optima / mild overfitting.

  ![Part (a): Initialization Sensitivity](hw4/output/prob6_a_initialization.png)
  ![Part (b): Varying Components](hw4/output/prob6_b_components.png)

- **File Structure for HW4:**
  - `hw4/hw4_solution.py`: Final implementation (GMM-EM training + experiments + plots)
  - `hw4/output/`:
    - `prob6_a_initialization.png` (25 init-pairs for \(C=8\))
    - `prob6_b_components.png` (PoE vs \(d\) for varying \(C\))
