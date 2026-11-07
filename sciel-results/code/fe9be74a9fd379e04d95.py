# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
        """Hybrid optimizer: Local init + Global DE + Local Polish."""

        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 270  # Safety margin
            D = self.dim
            budget = self.budget

            call_count = [0]
            best_x = [None]
            best_f = [np.inf]

            def track(x):
                call_count[0] += 1
                if call_count[0] > budget:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline:
                    raise TimeoutError("Time limit exceeded")
                x_arr = np.asarray(x, dtype=np.float64)
                x_arr = np.clip(x_arr, -5.0, 5.0)
                f_val = func(x_arr)
                f_val = float(f_val)
                if np.isnan(f_val) or np.isinf(f_val):
                    f_val = 1e10
                if f_val < best_f[0]:
                    best_f[0] = f_val
                    best_x[0] = x_arr.copy()
                return f_val

            # Phase 1: Quick local search from center
            # This helps if the shift is small, providing early progress for AOCC
            if budget > 60:
                x_center = np.zeros(D)
                try:
                    minimize(
                        track,
                        x_center,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except (ValueError, TimeoutError):
                    pass
                except Exception:
                    pass

            # Phase 2: Global search with Differential Evolution
            # Use remaining budget for DE
            remaining = budget - call_count[0]
            if remaining > 50:
                bounds = [(-5.0, 5.0)] * D
                # Adjust popsize and maxiter to fit remaining budget
                # popsize * maxiter <= remaining
                popsize = 10
                maxiter = max(1, (remaining - 10) // popsize)
                
                try:
                    differential_evolution(
                        track,
                        bounds,
                        maxiter=maxiter,
                        popsize=popsize,
                        tol=1e-8,
                        seed=42,
                        polish=False
                    )
                except (ValueError, TimeoutError):
                    pass
                except Exception:
                    pass

            # Phase 3: Final local polish
            if best_x[0] is not None and call_count[0] < budget - 5:
                x_start = best_x[0]
                try:
                    minimize(
                        track,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except (ValueError, TimeoutError):
                    pass
                except Exception:
                    pass

            return (best_x[0] if best_x[0] is not None else np.zeros(D),
                    best_f[0] if best_f[0] < np.inf else 0.0)
    # EVOLVE-BLOCK-END
