import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logreg import LogisticRegressionScratch, LogRegConfig
from src.optim_gd import gradient_descent, GDConfig
from src.optim_newton import newton_method, NewtonConfig


def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def run_all_methods(X_train, y_train, theta0, model):
    f = lambda th: model.loss(X_train, y_train, th)
    g = lambda th: model.grad(X_train, y_train, th)
    H = lambda th: model.hessian(X_train, th)

    # Fixed-step GD baseline (tunable)
    gd_fixed_cfg = GDConfig(
        max_iter=8000,
        tol=1e-6,
        use_line_search=False,
        step_size=1e-2,
    )

    # GD + Armijo line search
    gd_ls_cfg = GDConfig(
        max_iter=2000,
        tol=1e-6,
        use_line_search=True,
        alpha0=1.0,
        beta=0.5,
        c=1e-4,
        max_backtracks=50,
    )

    # Newton + Armijo line search
    newton_ls_cfg = NewtonConfig(
        max_iter=50,
        tol=1e-8,
        use_line_search=True,
        alpha0=1.0,
        beta=0.5,
        c=1e-4,
        max_backtracks=50,
        damping=1e-8,
    )

    results = {}
    results["GD_fixed_lr=1e-2"] = gradient_descent(f, g, theta0, gd_fixed_cfg)
    results["GD_line_search"] = gradient_descent(f, g, theta0, gd_ls_cfg)
    results["Newton_line_search"] = newton_method(f, g, H, theta0, newton_ls_cfg)

    return results


def plot_loss(results):
    plt.figure()
    for name, res in results.items():
        plt.plot(res.history["loss"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Breast Cancer: Loss vs Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/loss_vs_iter.png", dpi=200)
    plt.close()


def plot_grad_norm(results):
    plt.figure()
    for name, res in results.items():
        plt.semilogy(res.history["grad_norm"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("||grad|| (log scale)")
    plt.title("Breast Cancer: Gradient Norm vs Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/gradnorm_vs_iter.png", dpi=200)
    plt.close()


def plot_alpha(results):
    plt.figure()
    for name, res in results.items():
        if "alpha" in res.history and len(res.history["alpha"]) > 0:
            plt.plot(res.history["alpha"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("alpha")
    plt.title("Breast Cancer: Step Size (alpha) vs Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/alpha_vs_iter.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target.astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    # Scaling (IMPORTANT for optimization experiments)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegressionScratch(LogRegConfig(l2=1e-2, fit_intercept=True))
    theta0 = model.initialize(d=X_train.shape[1], seed=0)

    # Run
    results = run_all_methods(X_train, y_train, theta0, model)

    # Evaluate + print summary
    summary = {}
    print("\n=== Breast Cancer Results (test split) ===")
    print(f"{'Method':22s} | {'conv':4s} | {'iters':5s} | {'test_acc':8s} | {'fev':5s} | {'gev':5s}")
    print("-" * 72)

    for name, res in results.items():
        yhat = model.predict(X_test, res.x)
        acc = accuracy(y_test, yhat)
        iters = len(res.history["loss"])

        # Newton result has n_hev; GD doesn't. We'll handle safely.
        n_hev = getattr(res, "n_hev", None)

        print(f"{name:22s} | {str(res.converged):4s} | {iters:5d} | {acc:8.4f} | {res.n_fev:5d} | {res.n_gev:5d}")

        summary[name] = {
            "converged": bool(res.converged),
            "iters": int(iters),
            "test_accuracy": float(acc),
            "final_loss": float(res.history["loss"][-1]),
            "final_grad_norm": float(res.history["grad_norm"][-1]),
            "n_fev": int(res.n_fev),
            "n_gev": int(res.n_gev),
            "n_hev": None if n_hev is None else int(n_hev),
        }

    # Save summary json
    with open("results/tables/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Make plots
    plot_loss(results)
    plot_grad_norm(results)
    plot_alpha(results)

    print("\nSaved plots to:")
    print("  results/plots/loss_vs_iter.png")
    print("  results/plots/gradnorm_vs_iter.png")
    print("  results/plots/alpha_vs_iter.png")
    print("Saved summary to:")
    print("  results/tables/summary.json")


if __name__ == "__main__":
    main()
