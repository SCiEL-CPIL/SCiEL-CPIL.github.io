"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-scale CMA-ES variant with fixed step-size sequence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Fixed sequence of step sizes: Geometric decay from 4.0 to 1e-4
        # This is a deterministic CMA-ES-like strategy where we shrink the
        # search radius geometrically and restart only upon plateau detection.
        scales = np.geomspace(4.0, 1e-4, num=20)

        # Deterministic initial points: corners and center
        init_points = [
            np.zeros(dim),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
        ]

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy:
        # For each initial point, perform a deterministic multi-scale descent.
        # At each scale, we try moves in all dimensions (positive and negative)
        # and also some diagonal moves to handle correlated minima.
        # If no improvement is found at a scale, we shrink the scale (geometric decay).
        # We restart from a new initial point if we hit a plateau (no improvement for 3 consecutive scales).

        for init in init_points:
            if calls_made >= budget or time.time() > deadline:
                break

            x = init.copy()
            fx = evaluate(x)

            # Multi-scale descent
            no_improve_count = 0
            for scale_idx, scale in enumerate(scales):
                if calls_made >= budget or time.time() > deadline:
                    break

                improved = False

                # Try moves in each dimension
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Positive move
                    x_trial = x.copy()
                    x_trial[d] += scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = evaluate(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True

                    # Negative move
                    x_trial = x.copy()
                    x_trial[d] -= scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = evaluate(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True

                # Try some diagonal moves (pairs of dimensions)
                if dim >= 2:
                    for d1 in range(dim):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        for d2 in range(d1 + 1, dim):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            # Positive-positive diagonal
                            x_trial = x.copy()
                            x_trial[d1] += scale
                            x_trial[d2] += scale
                            x_trial = np.clip(x_trial, -5, 5)
                            if not np.allclose(x_trial, x):
                                f_trial = evaluate(x_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    improved = True

                            # Positive-negative diagonal
                            x_trial = x.copy()
                            x_trial[d1] += scale
                            x_trial[d2] -= scale
                            x_trial = np.clip(x_trial, -5, 5)
                            if not np.allclose(x_trial, x):
                                f_trial = evaluate(x_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    improved = True

                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If we've had 3 consecutive scales with no improvement,
                    # we're likely at a local minimum. Break to restart.
                    if no_improve_count >= 3:
                        break

            # If we still have budget, we could do a final polish with scipy
            # but that might be overkill. The geometric decay should be sufficient.

        # Final polish with scipy if we have budget left
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline:
            try:
                result = minimize(func, best_x, method='Nelder-Mead',
                                  options={'maxiter': budget - calls_made, 'xatol': 1e-6, 'fatol': 1e-8})
                calls_made += result.nfev
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
