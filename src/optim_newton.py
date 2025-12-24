# src/optim_newton.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np

from .line_search import backtracking_armijo


Array = np.ndarray


@dataclass
class NewtonConfig:
    max_iter: int = 50
    tol: float = 1e-8
    use_line_search: bool = True
    alpha0: float = 1.0
    beta: float = 0.5
    c: float = 1e-4
    max_backtracks: int = 50
    damping: float = 1e-8  # add to diagonal if Hessian is near-singular


@dataclass
class OptimizeResult:
    x: Array
    history: Dict[str, List[float]]
    n_fev: int
    n_gev: int
    n_hev: int
    converged: bool
    message: str


def newton_method(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    hess: Callable[[Array], Array],
    x0: Array,
    cfg: NewtonConfig,
) -> OptimizeResult:
    """
    Newton's method with optional Armijo backtracking line search.
    Includes basic damping for numerical stability.
    """
    x = x0.copy()
    history: Dict[str, List[float]] = {
        "loss": [],
        "grad_norm": [],
        "alpha": [],
    }

    n_fev = 0
    n_gev = 0
    n_hev = 0

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
                n_hev=n_hev,
                converged=True,
                message=f"Converged: ||grad|| <= tol at iter {k}",
            )

        H = hess(x); n_hev += 1

        # Damping to avoid singularity; also helps if Hessian is poorly conditioned
        H_damped = H + cfg.damping * np.eye(H.shape[0])

        # Solve H p = -g
        try:
            p = np.linalg.solve(H_damped, -g)
        except np.linalg.LinAlgError:
            # fallback: least squares solve
            p, *_ = np.linalg.lstsq(H_damped, -g, rcond=None)

        # Ensure descent direction; if not, fallback to steepest descent direction
        if float(g @ p) >= 0:
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
                history["alpha"].append(alpha)
                x = x + alpha * p
                f_x = float(f(x)); n_fev += 1
                return OptimizeResult(
                    x=x,
                    history=history,
                    n_fev=n_fev,
                    n_gev=n_gev,
                    n_hev=n_hev,
                    converged=False,
                    message=f"Line search failed at iter {k}: {ls.message}",
                )
        else:
            alpha = 1.0  # standard Newton

        history["alpha"].append(float(alpha))
        x = x + alpha * p
        f_x = float(f(x)); n_fev += 1

    return OptimizeResult(
        x=x,
        history=history,
        n_fev=n_fev,
        n_gev=n_gev,
        n_hev=n_hev,
        converged=False,
        message="Reached max_iter without convergence",
    )
