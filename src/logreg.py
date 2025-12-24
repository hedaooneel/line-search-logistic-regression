# src/logreg.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


Array = np.ndarray


def sigmoid(z: Array) -> Array:
    # Numerically stable sigmoid
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogRegConfig:
    l2: float = 0.0         # L2 regularization strength (lambda)
    fit_intercept: bool = True


class LogisticRegressionScratch:
    """
    Binary logistic regression with L2 regularization, implemented from scratch.

    Parameter vector is stored as a single 1D vector theta:
      - If fit_intercept=True: theta = [w (d,), b (1,)] of size d+1
      - Else: theta = [w (d,)] of size d
    """

    def __init__(self, config: LogRegConfig):
        self.config = config
        self.theta: Array | None = None

    def _unpack(self, theta: Array) -> Tuple[Array, float]:
        if self.config.fit_intercept:
            w = theta[:-1]
            b = float(theta[-1])
            return w, b
        return theta, 0.0

    def _pack(self, w: Array, b: float) -> Array:
        if self.config.fit_intercept:
            return np.concatenate([w, np.array([b], dtype=float)])
        return w.copy()

    def initialize(self, d: int, seed: int = 0) -> Array:
        rng = np.random.default_rng(seed)
        w0 = rng.normal(0.0, 0.01, size=d)
        b0 = 0.0
        self.theta = self._pack(w0, b0)
        return self.theta.copy()

    def predict_proba(self, X: Array, theta: Array) -> Array:
        w, b = self._unpack(theta)
        z = X @ w + b
        return sigmoid(z)

    def predict(self, X: Array, theta: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X, theta) >= threshold).astype(int)

    def loss(self, X: Array, y: Array, theta: Array) -> float:
        """
        Negative log-likelihood + (lambda/2)||w||^2
        """
        y = y.astype(float)
        p = self.predict_proba(X, theta)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)

        n = X.shape[0]
        data_term = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

        w, _ = self._unpack(theta)
        reg = 0.5 * self.config.l2 * float(w @ w)

        return float(data_term + reg)

    def grad(self, X: Array, y: Array, theta: Array) -> Array:
        """
        Gradient of loss wrt theta.
        """
        y = y.astype(float)
        n = X.shape[0]
        p = self.predict_proba(X, theta)

        w, _ = self._unpack(theta)

        # grad wrt w
        g_w = (X.T @ (p - y)) / n + self.config.l2 * w

        if self.config.fit_intercept:
            g_b = float((p - y).mean())
            return self._pack(g_w, g_b)

        return g_w

    def hessian(self, X: Array, theta: Array) -> Array:
        """
        Hessian wrt theta for Newton's method.
        For fit_intercept=True, returns (d+1) x (d+1) matrix.
        """
        n, d = X.shape
        p = self.predict_proba(X, theta)
        r = p * (1 - p)  # shape (n,)

        # Compute X^T R X efficiently: X^T (diag(r)) X
        XR = X * r[:, None]  # (n,d)
        H_ww = (X.T @ XR) / n

        # Add L2 to w-w block
        H_ww = H_ww + self.config.l2 * np.eye(d)

        if not self.config.fit_intercept:
            return H_ww

        # Blocks involving intercept
        H_wb = (X.T @ r) / n               # (d,)
        H_bb = float(r.mean())             # scalar

        H = np.zeros((d + 1, d + 1), dtype=float)
        H[:-1, :-1] = H_ww
        H[:-1, -1] = H_wb
        H[-1, :-1] = H_wb
        H[-1, -1] = H_bb
        return H
