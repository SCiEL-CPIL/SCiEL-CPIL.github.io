"""Direct solver for BBOB f9 (rotated Rosenbrock) exploiting valley structure."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Rosenbrock-aware start points and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Rosenbrock valley: points near the diagonal (1,1,...,1) are promising
        # Generate start points that sample the valley structure
        restarts = []

        # Points along the diagonal (the Rosenbrock valley)
        for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            p = np.full(dim, r)
            restarts.append(p)
            p_neg = np.full(dim, -r)
            restarts.append(p_neg)

        # Points near the diagonal with small perturbations
        for _ in range(10):
            base = rng.uniform(0.5, 3.5)
            p = np.full(dim, base) + rng.normal(0, 0.5, dim)
            restarts.append(np.clip(p, -5, 5))

        # Random points to explore the space
        for _ in range(10):
            p = rng.uniform(-4.5, 4.5, dim)
            restarts.append(p)

        # Axis-aligned points
        for axis in range(dim):
            for val in [-2.0, 2.0]:
                p = np.zeros(dim)
                p[axis] = val
                restarts.append(p)

        # Shuffle
        indices = list(range(len(restarts)))
        rng.shuffle(indices)
        restarts = [restarts[i] for i in indices]

        # Run L-BFGS-B from each start point
        for x0 in restarts:
            if time.time() > deadline:
                break
            remaining = budget - calls
            if remaining < 10:
                break

            alloc = min(remaining, max(20, budget // 10))
            if alloc < 10:
                break

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': max(1, alloc),
                        'maxfun': max(1, alloc),
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                continue

            if time.time() > deadline:
                break

            remaining = budget - calls
            if remaining < 5:
                break

            # Polish pass
            x_ref = result.x
            f_ref = result.fun
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

            if remaining >= 5:
                try:
                    result2 = minimize(
                        wrapped_func,
                        x_ref,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': max(1, remaining),
                            'maxfun': max(1, remaining),
                            'gtol': 1e-10,
                        }
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

            # Early termination if very close
            if best_f < 1e-8:
                break

        # Final polish
        remaining = budget - calls
        if remaining >= 3 and best_x is not None and time.time() < deadline:
            try:
                result3 = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': max(1, remaining),
                        'maxfun': max(1, remaining),
                        'gtol': 1e-12,
                    }
                )
                if result3.fun < best_f:
                    best_f = result3.fun
                    best_x = result3.x.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
