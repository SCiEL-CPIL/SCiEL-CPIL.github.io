import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # 330s limit, keep 45s margin for safety

        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() >= deadline:
                raise TimeoutError("Time limit exceeded")
            if eval_count >= budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            f_val = float(func(x))
            if np.isfinite(f_val):
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
            return f_val if np.isfinite(f_val) else 1e10

        # Phase 1: Differential Evolution for global exploration
        # Use a significant portion of the budget for DE
        de_budget = int(budget * 0.6)  # 300 evals for DE
        if de_budget >= 20 and time.time() < deadline:
            try:
                bounds = [(-5, 5)] * dim
                result = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    polish=False,
                    init='sobol',
                    maxfev=de_budget
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 2: Local polishing from the best point using Nelder-Mead
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            polish_budget = min(remaining, 100)
            if polish_budget >= 5:
                try:
                    minimize(
                        safe_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': polish_budget,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Phase 3: If still have budget, try another DE run with different seed
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining >= 50:
                try:
                    bounds = [(-5, 5)] * dim
                    result = differential_evolution(
                        safe_func,
                        bounds,
                        maxiter=50,
                        popsize=10,
                        tol=1e-8,
                        seed=123,
                        polish=False,
                        init='latinhypercube',
                        maxfev=remaining
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Final local polish if we have any budget left
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining >= 5:
                try:
                    minimize(
                        safe_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 30,
                            'maxfev': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Fallback: ensure at least one evaluation was made
        if eval_count == 0:
            rng = np.random.default_rng(42)
            for _ in range(min(10, budget)):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                try:
                    f_val = float(func(x))
                    eval_count += 1
                    if np.isfinite(f_val) and f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                except Exception:
                    break

        return best_x, best_f
