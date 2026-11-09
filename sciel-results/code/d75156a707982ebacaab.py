"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start L-BFGS-B Optimizer for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        call_count = [0]
        best_f = float('inf')
        best_x = None

        def elapsed():
            return time.time() - start_time

        def safe_func(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return float('inf')
            return val

        def objective(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            return safe_func(x)

        def run_lbfgsb(x_start, maxiter=500):
            nonlocal best_f, best_x
            bounds = [(-5, 5)] * n
            try:
                res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # Use Latin hypercube for good coverage
        n_starts = 12
        X_starts = np.zeros((n_starts, n))
        for d in range(n):
            perm = rng.permutation(n_starts)
            X_starts[:, d] = (perm + rng.uniform(size=n_starts)) / n_starts
        X_starts = X_starts * 10.0 - 5.0

        # Add a few more strategic points
        strategic = np.array([
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.array([1.0] * n),
            np.array([-1.0] * n),
        ])
        X_starts = np.vstack([X_starts, strategic[:n_starts - len(strategic)] if len(strategic) > n_starts else strategic])
        X_starts = X_starts[:n_starts]

        # Evaluate initial points and track best
        for x0 in X_starts:
            if call_count[0] >= budget - 50 or elapsed() > deadline - 60:
                break
            f0 = safe_func(x0)
            if np.isfinite(f0) and f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_func(best_x)

        # Phase 2: Local refinement from the best initial point
        if call_count[0] < budget - 20 and elapsed() < deadline - 30:
            run_lbfgsb(best_x, maxiter=800)

        # Phase 3: Multi-restart refinement from perturbed points
        n_restarts = 8
        for _ in range(n_restarts):
            if call_count[0] >= budget - 10 or elapsed() > deadline - 15:
                break
            # Perturb the current best
            perturbation = rng.normal(0, 0.5, size=n)
            x_trial = np.clip(best_x + perturbation, -5, 5)
            run_lbfgsb(x_trial, maxiter=300)

        # Phase 4: Final high-precision polish
        if call_count[0] < budget - 5 and elapsed() < deadline - 10:
            run_lbfgsb(best_x, maxiter=1000)

        return best_x, best_f
# EVOLVE-BLOCK-END
