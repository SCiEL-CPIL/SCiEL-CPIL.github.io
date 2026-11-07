import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300 - 20

        calls_made = 0
        best_x = np.random.default_rng(123).uniform(-5, 5, self.dim)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f
            if calls_made >= self.budget:
                return best_f
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Global Search with Differential Evolution
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * self.dim,
                popsize=14,
                maxiter=20,
                seed=42,
                tol=1e-7,
                init='latinhypercube',
                update_interval=1
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x)
        except Exception:
            pass

        # Phase 2: Local Polish with L-BFGS-B
        try:
            res_lbfgsb = minimize(
                wrapper,
                best_x,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 150, 'gtol': 1e-12}
            )
            if res_lbfgsb.fun < best_f:
                best_f = res_lbfgsb.fun
                best_x = np.array(res_lbfgsb.x)
        except Exception:
            pass

        # Phase 3: Multi-start Nelder-Mead refinement
        # Generate several starting points: best found, plus perturbations
        if calls_made < self.budget - 30:
            starts = [best_x.copy()]
            # Add a few random perturbations of best_x
            for i in range(3):
                perturb = best_x + rng.normal(0, 0.5, self.dim)
                perturb = np.clip(perturb, -5, 5)
                starts.append(perturb)

            for start in starts:
                if calls_made >= self.budget - 10:
                    break
                if time.time() - start_time > time_limit - 10:
                    break
                try:
                    res_nm = minimize(
                        wrapper,
                        start,
                        method='Nelder-Mead',
                        options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = np.array(res_nm.x)
                except Exception:
                    pass

        return best_x, best_f
