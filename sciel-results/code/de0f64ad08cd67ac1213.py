"""Multi-scale L-BFGS-B optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-scale L-BFGS-B with deterministic and random restarts for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(7)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_arr = np.asarray(x, dtype=float).ravel()
                if x_arr.shape[0] != dim:
                    x_arr = np.clip(x_arr[:dim], -5.0, 5.0)
                else:
                    x_arr = np.clip(x_arr, -5.0, 5.0)
                val = float(func(x_arr))
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x_arr.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        def lbfgsb_from(x0, max_iter=300):
            """Run L-BFGS-B from x0, tracking all evaluations."""
            if not time_ok():
                return x0, best_f

            def wrapped(x):
                if not time_ok():
                    return best_f
                return eval_and_track(x)

            try:
                result = minimize(
                    wrapped, x0, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max_iter, 'maxfun': max(10, budget - calls),
                             'gtol': 1e-6, 'ftol': 1e-10}
                )
                if result.x is not None:
                    xf = np.clip(np.asarray(result.x, dtype=float), -5.0, 5.0)
                    eval_and_track(xf)
                    return xf, best_f
            except Exception:
                pass
            return x0, best_f

        # Phase 1: Deterministic structured points for early AOCC
        # Center
        eval_and_track(np.zeros(dim))

        # Axis-aligned points at multiple scales
        for scale in [4.0, 3.0, 2.0, 1.0, 0.5]:
            for d in range(dim):
                if not time_ok():
                    break
                for sign in [-1, 1]:
                    x = np.zeros(dim)
                    x[d] = sign * scale
                    eval_and_track(x)

        # Face-center points
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                if not time_ok():
                    break
                x = np.zeros(dim)
                x[d1] = 3.0
                x[d2] = 3.0
                eval_and_track(x)
                x2 = np.zeros(dim)
                x2[d1] = -3.0
                x2[d2] = -3.0
                eval_and_track(x2)

        if not time_ok():
            return best_x, best_f

        # Phase 2: L-BFGS-B from the best point found so far
        if time_ok():
            lbfgsb_from(best_x.copy(), max_iter=400)

        if not time_ok():
            return best_x, best_f

        # Phase 3: Multi-scale random restarts with L-BFGS-B
        # Scales from large to small for exploration -> exploitation
        scales = [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
        max_restarts = 40
        restart_count = 0

        for scale in scales:
            if not time_ok():
                break
            # Number of restarts at this scale
            n_restarts = max(1, min(5, (budget - calls) // 15))
            for _ in range(n_restarts):
                if not time_ok():
                    break
                # Mix of random and structured perturbations
                if rng.random() < 0.5:
                    x0 = best_x + rng.normal(0, scale, size=dim)
                else:
                    # Structured: perturb a random subset of coordinates
                    x0 = best_x.copy()
                    n_perturb = rng.integers(1, dim + 1)
                    axes = rng.choice(dim, size=n_perturb, replace=False)
                    for ax in axes:
                        x0[ax] = rng.uniform(-5.0, 5.0)
                x0 = np.clip(x0, -5.0, 5.0)
                lbfgsb_from(x0, max_iter=200)
                restart_count += 1

        if not time_ok():
            return best_x, best_f

        # Phase 4: Fine-tuning - very local L-BFGS-B from best
        for _ in range(5):
            if not time_ok():
                break
            x0 = best_x + rng.normal(0, 0.05, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)
            lbfgsb_from(x0, max_iter=300)

        if not time_ok():
            return best_x, best_f

        # Phase 5: Final polish - one more L-BFGS-B from best with high maxiter
        if time_ok():
            lbfgsb_from(best_x.copy(), max_iter=500)

        return best_x, best_f
# EVOLVE-BLOCK-END
