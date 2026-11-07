"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time guard and budget tracking."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Time guard: kill after 300s wall-clock (330s hard limit)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin

        # Counting wrapper
        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                val = 1e10
            return val

        # Track best
        best_x = None
        best_f = np.inf

        # Generate diverse starting points
        # Mix of random, coordinate-axis, and structured starts
        starts = []

        # 1. Origin
        starts.append(np.zeros(dim))

        # 2. Random points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))

        # 3. Points near boundaries
        for _ in range(5):
            x = np.zeros(dim)
            for d in range(dim):
                if rng.random() < 0.5:
                    x[d] = rng.choice([-4.5, 4.5])
                else:
                    x[d] = rng.uniform(-2, 2)
            starts.append(x)

        # 4. Coordinate-axis points
        for d in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[d] = sign * 3.0
                starts.append(x)

        # Shuffle for diversity
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        for start in starts:
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            # Evaluate start
            fs = counting_func(start)
            if fs < best_f:
                best_f = fs
                best_x = start.copy()

            # Run L-BFGS-B from this start
            try:
                res = minimize(
                    counting_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 100, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

        # If we have budget left, do a final (1+1)-ES polish
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while call_count[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = counting_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
