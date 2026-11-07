import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """
    Grid-based multi-start optimizer for BBOB f9 (rotated Rosenbrock).
    Phase 1: Evaluate a 3^5 = 243 point grid to identify promising basins.
    Phase 2: Run Nelder-Mead from the top 5 grid points for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check
            if time.time() > deadline:
                return best_f
            # Budget check
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Deterministic 3^5 grid evaluation
        # Grid points: -5, 0, 5 in each dimension
        grid_levels = np.array([-5.0, 0.0, 5.0])
        grid_points = np.array(np.meshgrid(*[grid_levels] * dim)).T.reshape(-1, dim)
        
        # Evaluate all grid points and record their fitness
        grid_results = []
        for i, point in enumerate(grid_points):
            if calls_made >= budget or time.time() > deadline - 5.0:
                break
            f_val = wrapper(point)
            grid_results.append((f_val, point.copy()))
        
        # Sort by fitness and take top 5
        grid_results.sort(key=lambda x: x[0])
        top_starts = [point for _, point in grid_results[:5]]
        
        # If no valid results (shouldn't happen), fallback to origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)
            top_starts = [best_x.copy()]

        # Phase 2: Nelder-Mead local optimization from top 5 grid points
        # Allocate remaining budget across the 5 starts
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 5.0:
            # Distribute budget: give more to the best start
            # Start 1 gets 40%, starts 2-5 get 15% each
            nm_maxiters = [
                max(10, int(remaining_budget * 0.4)),
                max(5, int(remaining_budget * 0.15)),
                max(5, int(remaining_budget * 0.15)),
                max(5, int(remaining_budget * 0.15)),
                max(5, int(remaining_budget * 0.15)),
            ]
            
            for start_idx, start_point in enumerate(top_starts):
                if calls_made >= budget or time.time() > deadline - 3.0:
                    break
                try:
                    minimize(
                        wrapper,
                        start_point,
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_maxiters[start_idx],
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except Exception:
                    break

        # Phase 3: Final polish with L-BFGS-B from the best point if budget remains
        if calls_made < budget - 5 and time.time() < deadline - 2.0:
            try:
                minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f
