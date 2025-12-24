# Line Search Optimization for Logistic Regression
**Faster and More Stable Training Without Manual Learning Rate Tuning**

---

## Overview

Gradient-based machine learning models are often trained using fixed or heuristically tuned learning rates. This can lead to:

- Slow convergence  
- Numerical instability  
- Sensitivity to feature scaling and hyperparameters  

This project implements **binary logistic regression from scratch** and demonstrates how **line search optimization methods** automatically select step sizes that ensure stable and efficient training.

The following optimization strategies are compared:

- Gradient Descent with a fixed step size  
- Gradient Descent with backtracking line search (Armijo condition)  
- Newton’s Method with backtracking line search  

Across multiple datasets and experimental settings, line search consistently improves robustness and removes the need for manual learning-rate tuning.

---

## Key Contributions

- Implemented binary logistic regression from scratch (loss, gradient, prediction)
- Implemented backtracking line search using the **Armijo sufficient decrease condition**
- Integrated line search into:
  - Gradient Descent
  - Newton’s Method
- Empirical analysis of:
  - Convergence speed
  - Step-size behavior
  - Sensitivity to feature scaling
  - Stability across regularization strengths

---

## Optimization Methods

### Logistic Regression Objective

For data points $\((x_i, y_i)\)$ with $\(y_i \in \{0,1\}\)$, the objective function minimized is:

$$\[
f(w, b) =
-\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]$$
+ $$\frac{\lambda}{2}\|w\|_2^2
\]$$

where:

$$\[
p_i = \sigma(w^T x_i + b)
\]$$

and $$\(\sigma(\cdot)\)$$ is the sigmoid function.

---

### Backtracking Line Search (Armijo Condition)

At each iteration, the step size $\(\alpha_k\)$ is chosen to satisfy:

$$\[
f(x_k + \alpha_k p_k)
\le
f(x_k) + c\,\alpha_k \nabla f(x_k)^T p_k
\]$$

with parameters:

- Initial step size: $\(\alpha_0 = 1\) $ 
- Shrinkage factor: $\(\beta \in (0,1)\) $ 
- Sufficient decrease constant: $\(c \in (0,1)\)$ 

This guarantees descent whenever the search direction is valid and removes the need for manual learning-rate tuning.

---

## Methods Compared

| Method            | Step Size Strategy | Notes |
|------------------|-------------------|-------|
| Gradient Descent | Fixed              | Sensitive to tuning and scaling |
| Gradient Descent | Line Search        | Adaptive, stable, globally convergent |
| Newton’s Method  | Line Search        | Rapid local convergence, stable globally |

---

## Experiments

### Convergence Analysis
- Loss vs iteration  
- Gradient norm vs iteration  
- Step size $\(\alpha_k\)$ vs iteration  

### Feature Scaling Sensitivity
- Standardized vs unscaled features  
- Comparison of convergence stability  

### Regularization Sensitivity
- Multiple $\(\lambda\)$ values  
- Impact on curvature and step size  

### Performance Metrics
- Accuracy and AUC on test data  
- Runtime and number of function evaluations  

---

## Results (Summary)

- Fixed learning rates require careful tuning and may diverge on poorly scaled data  
- Line search automatically adapts step sizes, leading to stable convergence across datasets  
- Newton’s method with line search converges rapidly near the optimum while remaining stable far from it  
- Line search significantly reduces sensitivity to preprocessing and hyperparameter choice  

---

## Project Structure

```text
line-search-logreg/
├── README.md
├── src/
│   ├── logreg.py              # loss, gradient, prediction
│   ├── line_search.py         # Armijo backtracking
│   ├── optim_gd.py            # gradient descent
│   ├── optim_newton.py        # Newton's method
│   └── utils.py               # metrics and preprocessing
├── notebooks/
│   ├── convergence_analysis.ipynb
│   └── scaling_stability.ipynb
├── results/
│   ├── plots/
│   └── tables/
└── report.pdf
```

---

## How to Run

```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

Jupyter notebooks in the notebooks/ directory reproduce all plots and tables from the report.


---

## Why This Project Matters

Most machine learning pipelines treat optimization as a black box.

This project demonstrates a first-principles understanding of optimization, showing how line search methods improve reliability, interpretability, and performance in real machine learning workflows.


---

## Future Extensions

- Strong Wolfe line search
- Quasi-Newton methods (BFGS)
- Multiclass logistic regression
- Line search–based training of shallow neural networks

---

## Author

Neel Hedaoo
BS Computational Modeling & Data Analytics, Virginia Tech
