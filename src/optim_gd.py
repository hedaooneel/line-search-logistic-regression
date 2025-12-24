# src/optim_gd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

from .line_search import backtracking_armijo


Array = np.ndarray


@dataclass
class GDConfig:
    max_iter: int = 500
    tol: float = 1e-6
    step_size: float = 1e-2          # used if line_search=False
    use_line_search: bool = True
    alpha0: float = 1.0
    beta: float = 0.5
    c: float = 1e-4
    max_backtracks: int = 50


@dataclass
class OptimizeResult:
    x: Array
    history: Dict[str, List[float]]
    n_fev: int
    n_gev: int
    converged: bool
    message: str


def gradient_descent(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x0: Array,
    cfg: GDConfig,
) -> OptimizeResult:
    """
    Gradient descent with optional Armijo backtracking line search.
    Logs convergence info in `history`.
    """
    x = x0.copy()
    history: Dict[str, List[float]] = {
        "loss": [],
        "grad_norm": [],
        "alpha": [],
    }

    n_fev = 0
    n_gev = 0

    f_x = float(f(x)); n_fev += 1

    for k in range(cfg.max_iter):
        g = grad(x); n_gev += 1
        gnorm = float(np.linalg.norm(g))

        history["loss"].append(f_x)
        history["grad_norm"].append(gnorm)

        if gnorm <= cfg.tol:
            return OptimizeResult(
                x=x,
                history=history,
                n_fev=n_fev,
                n_gev=n_gev,
                converged=True,
                message=f"Converged: ||grad|| <= tol at iter {k}",
            )

        p = -g

        if cfg.use_line_search:
            ls = backtracking_armijo(
                f=f,
                grad=grad,
                x=x,
                p=p,
                alpha0=cfg.alpha0,
                beta=cfg.beta,
                c=cfg.c,
                max_backtracks=cfg.max_backtracks,
                f_x=f_x,
                g_x=g,
            )
            alpha = ls.alpha
            n_fev += ls.n_fev
            if not ls.success:
                # still take the best alpha we ended with, but mark as not converged if it stalls
                history["alpha"].append(alpha)
                x = x + alpha * p
                f_x = float(f(x)); n_fev += 1
                return OptimizeResult(
                    x=x,
                    history=history,
                    n_fev=n_fev,
                    n_gev=n_gev,
                    converged=False,
                    message=f"Line search failed at iter {k}: {ls.message}",
                )
        else:
            alpha = cfg.step_size

        history["alpha"].append(float(alpha))
        x = x + alpha * p
        f_x = float(f(x)); n_fev += 1

    return OptimizeResult(
        x=x,
        history=history,
        n_fev=n_fev,
        n_gev=n_gev,
        converged=False,
        message="Reached max_iter without convergence",
    )
