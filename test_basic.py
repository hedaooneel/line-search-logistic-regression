import numpy as np

from src.logreg import LogisticRegressionScratch, LogRegConfig
from src.optim_gd import gradient_descent, GDConfig
from src.optim_newton import newton_method, NewtonConfig


def make_synthetic(n=500, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    true_b = 0.2
    logits = X @ true_w + true_b
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def main():
    X, y = make_synthetic()

    # Create model
    model = LogisticRegressionScratch(LogRegConfig(l2=1e-2, fit_intercept=True))
    theta0 = model.initialize(d=X.shape[1], seed=0)

    # Define function handles for optimizer
    f = lambda th: model.loss(X, y, th)
    g = lambda th: model.grad(X, y, th)
    H = lambda th: model.hessian(X, th)

    # ---- Gradient Descent + Line Search ----
    gd_cfg = GDConfig(max_iter=300, tol=1e-6, use_line_search=True)
    gd_res = gradient_descent(f, g, theta0, gd_cfg)

    print("\n=== Gradient Descent (Line Search) ===")
    print("Converged:", gd_res.converged)
    print("Message:", gd_res.message)
    print("Final loss:", gd_res.history["loss"][-1])
    print("Final grad norm:", gd_res.history["grad_norm"][-1])
    print("Last alpha:", gd_res.history["alpha"][-1])
    print("Function evals:", gd_res.n_fev, "Grad evals:", gd_res.n_gev)

    # ---- Newton + Line Search ----
    newton_cfg = NewtonConfig(max_iter=50, tol=1e-8, use_line_search=True)
    newton_res = newton_method(f, g, H, theta0, newton_cfg)

    print("\n=== Newton (Line Search) ===")
    print("Converged:", newton_res.converged)
    print("Message:", newton_res.message)
    print("Final loss:", newton_res.history["loss"][-1])
    print("Final grad norm:", newton_res.history["grad_norm"][-1])
    print("Last alpha:", newton_res.history["alpha"][-1])
    print(
        "Function evals:", newton_res.n_fev,
        "Grad evals:", newton_res.n_gev,
        "Hess evals:", newton_res.n_hev
    )

    # Simple accuracy check
    yhat = model.predict(X, newton_res.x)
    acc = (yhat == y).mean()
    print("\nTraining accuracy (Newton):", round(float(acc), 4))


if __name__ == "__main__":
    main()
