"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-restart CMA-ES."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES with adaptive budgets and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        # Wrap func to count calls and enforce budget
        def counting_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # Adaptive restart schedule: total budget split across restarts.
        # Early restarts are cheap (exploration), later ones larger (exploitation).
        # We use a geometric-like split.
        # Number of restarts: start with a few, adapt based on remaining budget.
        # Strategy: first restart uses ~15% of budget, second ~20%, etc.,
        # with re-seeding near best point for later restarts.

        # Define restart fractions (cumulative) - sum to 1.0
        # We'll do up to 6 restarts with adaptive sizing.
        # Early restarts smaller to explore, later larger to exploit.
        n_restarts_max = 6
        # Fraction of budget per restart (roughly geometric increasing)
        # e.g., 0.08, 0.10, 0.14, 0.20, 0.26, 0.22 -> sums to 1.0
        fractions = np.array([0.08, 0.10, 0.14, 0.20, 0.26, 0.22], dtype=float)
        fractions = fractions / fractions.sum()  # normalize

        for r in range(n_restarts_max):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Budget for this restart
            restart_budget = int(budget * fractions[r])
            if restart_budget < 20:
                restart_budget = 20
            # Don't exceed remaining budget
            remaining = budget - call_count
            if restart_budget > remaining:
                restart_budget = remaining
            if restart_budget < 10:
                break

            # Choose initial point
            if r == 0:
                # First restart: sample broadly
                x0 = rng.uniform(-4.5, 4.5, dim)
            else:
                # Re-seed near global best with some perturbation
                if best_x is not None:
                    # Perturb around best point; scale perturbation with restart index
                    # Later restarts: smaller perturbation (more exploitation)
                    scale = 1.5 * (0.7 ** r)
                    x0 = best_x + scale * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                else:
                    x0 = rng.uniform(-4.5, 4.5, dim)

            # CMA-ES parameters
            sigma0 = 1.0 if r == 0 else 0.5 * (0.8 ** r)
            # Max iterations for this restart (CMA-ES internal)
            # Each CMA-ES iteration uses (lambda) function evaluations.
            # Default lambda for CMA-ES in scipy is ~ 4 + floor(3*log(n))
            # For n=5, lambda ~ 4 + 3*1 = 7 (approximately)
            # So maxiter ~ restart_budget / 7
            maxiter = max(5, restart_budget // 7)

            # Run CMA-ES
            try:
                res = minimize(
                    counting_func,
                    x0,
                    method='CMA-ES',
                    options={
                        'maxiter': maxiter,
                        'sigma0': sigma0,
                        'x0': x0,
                        'bounds': [(-5.0, 5.0)] * dim,
                        'disp': False,
                        'verbose': 0,
                        'tolfun': 1e-12,
                        'tolx': 1e-12,
                        'seed': int(rng.integers(0, 2**31)),
                    }
                )
                # Update global best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
            except Exception:
                # If CMA-ES fails, do a simple local search fallback
                if best_x is None:
                    best_x = np.clip(x0, -5.0, 5.0)
                    best_f = counting_func(best_x)
                # Try a few random steps
                for _ in range(min(20, budget - call_count)):
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5.0, 5.0)
                    fy = counting_func(y)
                    if fy < best_f:
                        best_f = fy
                        best_x = y

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
