import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 285  # Safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isnan(val) or np.isinf(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Deterministic coarse grid search over [-4, 4]^5
        # Use a 3x3x3x3x3 grid = 243 points to cover the likely basin
        grid_points = []
        grid_vals = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            if self.calls >= budget or time.time() > self.deadline:
                                break
                            x = np.array([i * 2.667 - 4.0, j * 2.667 - 4.0, 
                                          k * 2.667 - 4.0, l * 2.667 - 4.0,
                                          m * 2.667 - 4.0])
                            try:
                                f_val = wrapped_func(x)
                                grid_points.append(x)
                                grid_vals.append(f_val)
                            except (TimeoutError, ValueError):
                                break
                            except Exception:
                                continue

        # Also add origin and some strategic points
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
        ]
        for sp in strategic_points:
            if self.calls >= budget or time.time() > self.deadline:
                break
            try:
                wrapped_func(sp)
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 2: Nelder-Mead refinement from the best grid point
        if self.calls < budget - 10 and time.time() < self.deadline and self.best_x is not None:
            try:
                res_nm = minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = res_nm.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: L-BFGS-B from multiple promising starting points
        # Use top 5 grid points plus best found so far
        if self.calls < budget - 10 and time.time() < self.deadline:
            # Collect candidate starting points
            candidates = [self.best_x] if self.best_x is not None else []
            
            if grid_points and grid_vals:
                sorted_idx = np.argsort(grid_vals)
                for idx in sorted_idx[:5]:
                    if idx < len(grid_points):
                        candidates.append(grid_points[idx])
            
            # Add some random perturbations around best
            if self.best_x is not None:
                for _ in range(3):
                    perturbation = rng.normal(0, 0.3, dim)
                    new_start = np.clip(self.best_x + perturbation, -5, 5)
                    candidates.append(new_start)

            for start in candidates:
                if self.calls >= budget or time.time() > self.deadline:
                    break
                if start is None:
                    continue
                try:
                    res_lbfgs = minimize(
                        wrapped_func,
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
                    )
                    if res_lbfgs.fun < self.best_f:
                        self.best_f = res_lbfgs.fun
                        self.best_x = res_lbfgs.x.copy()
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # Phase 4: Final Nelder-Mead polish if budget remains
        if self.calls < budget - 5 and time.time() < self.deadline and self.best_x is not None:
            try:
                res_nm2 = minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm2.fun < self.best_f:
                    self.best_f = res_nm2.fun
                    self.best_x = res_nm2.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Fallback: if no valid result, evaluate origin
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            try:
                self.best_f = wrapped_func(self.best_x)
            except:
                self.best_f = 1e10

        return self.best_x, self.best_f
