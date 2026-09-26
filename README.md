\# AI-ML Recruitment Technical Submission - 2026

\*\*Candidate Name:\*\* Yeddula Sai Venkata Reddy  

\*\*Enrolment Track:\*\* Second Year Recruitment  

\*\*Domain:\*\* Artificial Intelligence \& Machine Learning  



\---



\## 1. Candidate Details

\* \*\*Name:\*\* Yeddula Sai Venkata Reddy

\* \*\*Branch/Year:\*\* Second Year B.Tech, SRMIST

\* \*\*Submission Track:\*\* AI-ML Recruitment Tasks (Second Year)



\---



\## 2. Tasks Completed

\* \*\*Task 1:\*\* Air Quality Forecasting (`AeroCast`) using the UCI Air Quality Dataset

\* \*\*Task 2:\*\* Neural Network Digit Classification (`GlyphNet`) using PyTorch and MNIST



\---



\## 3. Problem Statements

\* \*\*Task 1:\*\* Build a robust machine learning workflow to forecast atmospheric Benzene ($C\_6H\_6$) concentrations using multi-sensor environmental data, while addressing sensor corruption flags, modeling cyclical time dynamics, and strictly avoiding temporal data leakage.

\* \*\*Task 2:\*\* Architect and evaluate a deep neural network to classify handwritten digits (0–9) on MNIST, analyzing activation mechanisms, loss trajectories, confusion matrix patterns, and high-confidence edge-case failures.



\---



\## 4. Technical Approach \& Architecture



\### Task 1: Time-Series Forecasting (AeroCast)

\* \*\*Sentinel Value Imputation:\*\* UCI flags missing or corrupt sensor readings using `-200`. Dropping rows breaks consecutive time steps; instead, localized time-weighted spline interpolation (limit=3 hours) was applied to retain physical atmospheric continuity.

\* \*\*Harmonic Cyclical Projections:\*\* Hours of the day and calendar months were mapped to polar coordinates ($\\sin/\\cos$) to preserve natural circular transitions between 23:00 and 00:00.

\* \*\*Leakage-Free Feature Engineering:\*\* Created causal autoregressive lags ($t-1, t-2, t-3, t-24$) and rolling window statistics strictly shifted by `shift(1)`, guaranteeing step $t$ never sees step $t$.

\* \*\*Model Engine:\*\* Deployed an early-stopping LightGBM regressor evaluated strictly over an out-of-time chronological partition (first 80% train, last 20% test).



\### Task 2: Modular PyTorch Neural Network (GlyphNet)

\* \*\*Preprocessing:\*\* Scaled tensors using empirical MNIST population parameters ($\\mu = 0.1307, \\sigma = 0.3081$) to center gradients and accelerate AdamW convergence.

\* \*\*Network Topology:\*\* 

&#x20; $$\\text{Input}(784) \\to \\text{Linear}(256) \\to \\text{BatchNorm} \\to \\text{ReLU} \\to \\text{Dropout}(0.2) \\to \\text{Linear}(128) \\to \\text{BatchNorm} \\to \\text{ReLU} \\to \\text{Linear}(10)$$

\* \*\*Diagnostic Analytics:\*\* Extracted per-class precision/recall, generated a multi-class confusion matrix, and built an automated failure-mode gallery to isolate ambiguous stroke patterns.

\* \*\*Ablation Study:\*\* Conducted a controlled experiment evaluating standard ReLU against smooth Gaussian Error Linear Units (GELU).



\---



\## 5. Technologies Used

\* \*\*Languages \& Core:\*\* Python 3.10+, NumPy, Pandas, SciPy

\* \*\*Deep Learning Framework:\*\* PyTorch, TorchVision

\* \*\*Machine Learning:\*\* LightGBM, Scikit-Learn

\* \*\*Explainability \& Diagnostics:\*\* SHAP (Shapley Additive exPlanations)

\* \*\*Visualization:\*\* Matplotlib, Seaborn



\---



\## 6. Results \& Benchmark Metrics



\### Task 1: Air Quality Regression ($C\_6H\_6$ Concentration)

\* \*\*Mean Absolute Error (MAE):\*\* 0.4182 mg/m³

\* \*\*Mean Squared Error (MSE):\*\* 0.3811

\* \*\*Root Mean Squared Error (RMSE):\*\* 0.6173 mg/m³

\* \*\*$R^2$ Score:\*\* 0.9624



\### Task 2: MNIST Digit Classification

\* \*\*Test Accuracy:\*\* 98.42%

\* \*\*Macro Precision / Recall / F1:\*\* 0.9841 / 0.9840 / 0.9840

\* \*\*Test Loss:\*\* 0.0519



\---



\## 7. Key Learnings

1\. \*\*Temporal Horizon Integrity:\*\* Standard random shuffling introduces lookahead bias into time-series problems. Enforcing strict chronological boundaries is essential for realistic evaluation.

2\. \*\*Gradient Stability via Normalization:\*\* Standardizing image tensors using global population moments rather than naive min-max scaling prevents activation saturation and stabilizes weight updates.

3\. \*\*Failure Diagnostics Over Raw Accuracy:\*\* High global accuracy can mask distinct class confusions (e.g., stroke similarities between handwritten 4s and 9s). Systematic residual inspection exposes model blind spots.



\---



\## 8. Challenges \& Engineering Solutions

\* \*\*Challenge:\*\* Initial rolling window features produced artificially inflated $R^2$ scores because standard `.rolling()` includes the current time step's target value.

\* \*\*Solution:\*\* Applied `.shift(1)` across all aggregate rolling features. This enforced a strict boundary where predictions at time $t$ only have access to information up to $t-1$, ensuring an honest and robust out-of-time evaluation.

