"""Rewritten optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with gradient-free polish for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 260.0

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def scipy_wrapper(x):
            nonlocal call_count, deadline
            if call_count >= budget or time.time() > deadline:
                return best_f
            return safe_func(x)

        rng = np.random.default_rng(12345)

        # Phase 1: Sparse but structured initial sampling
        # For rotated Rosenbrock, the valley structure is rotated but still present.
        # We sample along the unrotated valley and add random perturbations.
        candidates = []

        # Core points: unrotated minimum and nearby
        candidates.append(np.ones(dim))
        candidates.append(np.zeros(dim))

        # Valley points: x_i = x_0^(2^i) - 1 - ... (approximate valley)
        # For Rosenbrock f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
        # Valley is roughly x_{i+1} = x_i^2, with x_0 near 1
        for x0 in [-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            c = np.zeros(dim)
            c[0] = x0
            for i in range(1, dim):
                c[i] = c[i-1]**2
            c = np.clip(c, -5, 5)
            candidates.append(c)

        # Diagonal points
        for val in [-3, -2, -1, 0, 1, 2, 3]:
            candidates.append(np.full(dim, val, dtype=np.float64))

        # Random points with some structure
        for _ in range(15):
            # Mix of uniform and slightly structured random
            if rng.random() < 0.5:
                c = rng.uniform(-4, 4, dim)
            else:
                # Perturbed valley
                x0 = rng.uniform(-2, 2)
                c = np.zeros(dim)
                c[0] = x0
                for i in range(1, dim):
                    c[i] = c[i-1]**2 + rng.normal(0, 0.5)
                c = np.clip(c, -5, 5)
            candidates.append(c)

        # Evaluate candidates
        candidate_results = []
        for c in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            c_clipped = np.clip(c, -5.0, 5.0)
            f = safe_func(c_clipped)
            candidate_results.append((f, c_clipped.copy()))

        if not candidate_results:
            c = np.zeros(dim)
            f = safe_func(c)
            candidate_results.append((f, c.copy()))

        candidate_results.sort(key=lambda x: x[0])

        # Phase 2: Multi-start L-BFGS-B
        remaining = budget - call_count
        n_starts = min(len(candidate_results), 10)

        # Allocate budget proportionally, giving more to better starts
        weights = np.array([np.exp(-i * 0.4) for i in range(n_starts)])
        weights /= weights.sum()

        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break

            f_init, x_init = candidate_results[i]
            maxfev = max(20, int(remaining * weights[i]))
            maxfev = min(maxfev, budget - call_count)

            if maxfev < 15:
                continue

            try:
                result = minimize(
                    scipy_wrapper,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': maxfev,
                        'maxfev': maxfev,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                    }
                )
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 3: Gradient-free polish with Nelder-Mead (adaptive)
        if call_count < budget - 40 and best_x is not None and time.time() < deadline:
            remaining = budget - call_count
            try:
                result = minimize(
                    scipy_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': remaining,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B polish
        if call_count < budget - 20 and best_x is not None and time.time() < deadline:
            remaining = budget - call_count
            try:
                result = minimize(
                    scipy_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': remaining,
                        'maxfev': remaining,
                        'gtol': 1e-12,
                        'ftol': 1e-15,
                    }
                )
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
