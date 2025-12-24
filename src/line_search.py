# src/line_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np


Array = np.ndarray


@dataclass
class LineSearchResult:
    alpha: float
    n_fev: int
    success: bool
    message: str


def backtracking_armijo(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x: Array,
    p: Array,
    alpha0: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    max_backtracks: int = 50,
    f_x: Optional[float] = None,
    g_x: Optional[Array] = None,
) -> LineSearchResult:
    """
    Backtracking line search enforcing the Armijo (sufficient decrease) condition:

        f(x + alpha p) <= f(x) + c * alpha * grad(x)^T p

    Parameters
    ----------
    f, grad : callables
        Objective and gradient. Both accept x as a 1D numpy array.
    x : np.ndarray
        Current iterate (shape: (d,))
    p : np.ndarray
        Search direction (shape: (d,))
    alpha0 : float
        Initial step size (often 1.0)
    beta : float
        Shrink factor in (0,1), e.g., 0.5
    c : float
        Armijo constant in (0,1), e.g., 1e-4
    max_backtracks : int
        Maximum number of shrink steps
    f_x : float, optional
        Precomputed f(x) to save an evaluation
    g_x : np.ndarray, optional
        Precomputed grad(x) to save an evaluation

    Returns
    -------
    LineSearchResult
    """
    if alpha0 <= 0:
        return LineSearchResult(0.0, 0, False, "alpha0 must be > 0")
    if not (0.0 < beta < 1.0):
        return LineSearchResult(0.0, 0, False, "beta must be in (0,1)")
    if not (0.0 < c < 1.0):
        return LineSearchResult(0.0, 0, False, "c must be in (0,1)")

    if f_x is None:
        f_x = float(f(x))
        n_fev = 1
    else:
        n_fev = 0

    if g_x is None:
        g_x = grad(x)

    # Ensure descent direction
    gtp = float(g_x @ p)
    if gtp >= 0:
        return LineSearchResult(0.0, n_fev, False, "p is not a descent direction (grad^T p >= 0)")

    alpha = float(alpha0)

    for _ in range(max_backtracks):
        x_new = x + alpha * p
        f_new = float(f(x_new))
        n_fev += 1

        if f_new <= f_x + c * alpha * gtp:
            return LineSearchResult(alpha, n_fev, True, "Armijo condition satisfied")

        alpha *= beta

    return LineSearchResult(alpha, n_fev, False, "Max backtracks reached without satisfying Armijo")
