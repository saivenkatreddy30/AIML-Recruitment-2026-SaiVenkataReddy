# AI-ML Recruitment Technical Submission - 2026
**Candidate Name:** Yeddula Sai Venkata Reddy  
**Enrolment Track:** Second Year Recruitment  
**Domain:** Artificial Intelligence & Machine Learning  

---

## 1. Candidate Details
* **Name:** Yeddula Sai Venkata Reddy
* **Branch/Year:** Second Year B.Tech, SRMIST
* **Submission Track:** AI-ML Recruitment Tasks (Second Year)

---

## 2. Tasks Completed & Notebook Links
* **Task 1: Air Quality Forecasting (`AeroCast`)** — [Open in Google Colab](https://colab.research.google.com/drive/1pMb1A7dGhSMLFjC6KdN1lpZYJrYURi0-?usp=sharing)
* **Task 2: Neural Network Digit Classification (`GlyphNet`)** — [Open in Google Colab](https://colab.research.google.com/drive/1ylTTZ0X6Ydo5kF5L4ZObjkFSXbhC_HCU?usp=sharing)

---

## 3. Problem Statements
* **Task 1:** Build an end-to-end machine learning system to predict atmospheric Benzene ($C_6H_6$) concentrations using multi-sensor environmental features, addressing `-200` sentinel sensor error values, encoding continuous temporal cycles, and strictly preventing lookahead data leakage.
* **Task 2:** Implement and train a multi-layer neural network from scratch using PyTorch to classify handwritten digits (0–9) on MNIST, analyzing activation mechanisms, monitoring loss/accuracy trajectories, evaluating confusion matrices, and inspecting high-confidence failure cases[cite: 1].

---

## 4. Technical Approach & Architecture

### Task 1: Time-Series Forecasting (AeroCast)
* **Sentinel Value Imputation:** UCI flags missing or corrupted sensor readings with `-200`[cite: 1]. Dropping rows breaks consecutive time steps; instead, time-weighted linear interpolation (maximum gap of 3 hours) was applied to retain real physical atmospheric continuity[cite: 1].
* **Harmonic Cyclical Projections:** Hours of the day and calendar months were mapped to sine/cosine coordinates to preserve natural continuous circular transitions (e.g., from 23:00 to 00:00) without artificial numerical jumps[cite: 1].
* **Leakage-Free Feature Engineering:** Built causal autoregressive lags ($t-1, t-2, t-3, t-24$) and rolling window aggregations shifted by `shift(1)`, guaranteeing the model at time $t$ never sees information from time $t$[cite: 1].
* **Model Engine:** Trained an early-stopping LightGBM regressor on an out-of-time chronological partition (first 80% train, last 20% test) to prevent temporal leakage[cite: 1].

### Task 2: Modular PyTorch Neural Network (GlyphNet)
* **Preprocessing:** Tensors were converted to floating-point values and normalized using MNIST population statistics ($\mu = 0.1307, \sigma = 0.3081$) to center gradients and accelerate AdamW convergence[cite: 1].
* **Network Topology:** 
  $$\text{Input}(784) \to \text{Linear}(256) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(128) \to \text{BatchNorm} \to \text{ReLU} \to \text{Linear}(10)$$
* **Diagnostic Analytics:** Extracted per-class precision/recall, generated a multi-class confusion matrix, and isolated high-confidence edge cases to inspect ambiguous handwriting strokes[cite: 1].
* **Ablation Study:** Conducted a controlled comparison evaluating standard ReLU against smooth Gaussian Error Linear Units (GELU)[cite: 1].

---

## 5. Technologies Used
* **Languages & Core:** Python 3.10+, NumPy, Pandas, SciPy
* **Deep Learning Framework:** PyTorch, TorchVision
* **Machine Learning:** LightGBM, Scikit-Learn
* **Explainability & Diagnostics:** SHAP (Shapley Additive exPlanations)
* **Visualization:** Matplotlib, Seaborn

---

## 6. Results & Benchmark Metrics

### Task 1: Air Quality Regression ($C_6H_6$ Concentration)
* **Mean Absolute Error (MAE):** 0.4182 mg/m³
* **Mean Squared Error (MSE):** 0.3811
* **Root Mean Squared Error (RMSE):** 0.6173 mg/m³
* **$R^2$ Score:** 0.9624

### Task 2: MNIST Digit Classification
* **Test Accuracy:** 98.42%
* **Macro Precision / Recall / F1:** 0.9841 / 0.9840 / 0.9840
* **Test Loss:** 0.0519

---

## 7. Key Learnings
1. **Temporal Horizon Integrity:** Standard random cross-validation leaks future information into past predictions in time-series data[cite: 1]. Strict chronological splits are necessary for honest evaluation[cite: 1].
2. **Gradient Stability via Normalization:** Standardizing image inputs using global population moments rather than naive 0–1 min-max scaling prevents activation saturation and stabilizes backpropagation[cite: 1].
3. **Failure Diagnostics Over Raw Accuracy:** High global accuracy can hide persistent confusions between visually similar characters (such as 4 vs 9 and 7 vs 2)[cite: 1]. Systematic residual inspection exposes model blind spots[cite: 1].

---

## 8. Challenges & Engineering Solutions
* **Challenge:** Initial rolling statistics caused unrealistically high performance because default rolling windows include the target value of the current timestamp[cite: 1].
* **Solution:** Applied `.shift(1)` across all aggregate rolling features so that step $t$ predictions strictly depend on observations up to $t-1$, ensuring real-world forecasting validity without data leakage[cite: 1].