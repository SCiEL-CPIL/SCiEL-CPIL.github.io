"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start L-BFGS-B with deadline guard for 5D Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        from scipy.optimize import minimize

        rng = np.random.default_rng(42)

        # Generate multiple starting points
        # Include origin, random points, and points along axes
        starts = []

        # Origin
        starts.append(np.zeros(self.dim))

        # Random uniform points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Points along each axis (positive and negative)
        for d in range(self.dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(self.dim)
                p[d] = val
                starts.append(p)

        # Diagonal points
        for val in [-3.0, 3.0]:
            starts.append(np.full(self.dim, val))

        # A few more structured points
        for _ in range(5):
            p = rng.uniform(-4, 4, self.dim)
            # Ensure at least one coordinate is reasonably large
            if np.max(np.abs(p)) < 1.0:
                p[rng.integers(0, self.dim)] = rng.choice([-3.0, 3.0])
            starts.append(p)

        # Shuffle starts to diversify
        indices = rng.permutation(len(starts))
        starts = [starts[i] for i in indices]

        for start in starts:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    eval_func,
                    start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
            except Exception:
                continue

            # Check if we got a valid result
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

            # If we're very close to optimal, do a final polish
            if best_f < 1e-10:
                break

            # If we've used most of our budget, stop
            if call_count > self.budget * 0.8:
                break

        # Final polish with tighter tolerances if we have budget left
        if best_x is not None and call_count < self.budget and time.time() < deadline:
            try:
                result = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
