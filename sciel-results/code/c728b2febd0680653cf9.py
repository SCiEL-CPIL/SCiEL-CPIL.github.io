"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Coarse lattice + Nelder-Mead refinement for Weierstrass f16.
    Phase 1: 3^5 grid (243 calls) for global coverage.
    Phase 2: Nelder-Mead local optimization from best grid point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Deterministic 3^5 lattice search
        grid_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            x = np.array([
                                -5.0 + i1 * 5.0,
                                -5.0 + i2 * 5.0,
                                -5.0 + i3 * 5.0,
                                -5.0 + i4 * 5.0,
                                -5.0 + i5 * 5.0
                            ])
                            grid_points.append(x)

        for x in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead refinement from best grid point
        x0 = best_x.copy() if best_x is not None else np.zeros(dim)
        
        # Wrap func to track calls and enforce deadline
        def wrapped_func(x):
            if time.time() > deadline:
                return np.inf
            return evaluate(x)

        # Run Nelder-Mead with tight tolerances for precision
        try:
            result = minimize(
                wrapped_func,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'maxiter': 5000,
                    'maxfev': budget - calls
                }
            )
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
