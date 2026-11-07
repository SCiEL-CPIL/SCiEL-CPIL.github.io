"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with CMA-ES refinement and deterministic seeding."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # Track calls and best
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def check_deadline():
            return time.time() > deadline or calls >= budget

        # Phase 1: Deterministic grid search for guaranteed early progress
        # Evaluate vertices of the hypercube and center
        vertices = [np.zeros(n)]
        for i in range(2 ** n):
            v = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
            vertices.append(v * 10.0 - 5.0)
        
        # Also add some intermediate points for better coverage
        for val in [-3.0, 0.0, 3.0]:
            for axis in range(n):
                x = np.zeros(n)
                x[axis] = val
                vertices.append(x)
        
        # Deduplicate
        unique_vertices = []
        seen = set()
        for v in vertices:
            key = tuple(np.round(v, 6))
            if key not in seen:
                seen.add(key)
                unique_vertices.append(v)
        
        for v in unique_vertices:
            if check_deadline():
                break
            safe_call(v)
        
        # Phase 2: Multi-start Nelder-Mead from best points
        # Sort evaluated points by function value and use top ones as starts
        if best_x is not None and not check_deadline():
            # Use the best point as primary start
            x0 = best_x.copy()
            
            def objective(x):
                if check_deadline():
                    return np.inf
                return safe_call(x)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'maxfev': 150,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Additional Nelder-Mead restarts from diverse points
        if not check_deadline() and best_x is not None:
            rng = np.random.default_rng(123)
            num_restarts = 5
            evals_per_restart = 60
            
            for r in range(num_restarts):
                if check_deadline():
                    break
                
                # Generate diverse start points around best_x
                if r == 0:
                    x0 = best_x.copy() + rng.normal(0, 1.0, n)
                elif r == 1:
                    x0 = best_x.copy() + rng.normal(0, 2.0, n)
                elif r == 2:
                    x0 = rng.uniform(-4.0, 4.0, n)
                elif r == 3:
                    x0 = best_x.copy()
                    x0[np.random.choice(n, 2, replace=False)] += rng.normal(0, 1.5, 2)
                else:
                    x0 = rng.uniform(-4.0, 4.0, n)
                
                x0 = np.clip(x0, -5.0, 5.0)
                
                def objective(x):
                    if check_deadline():
                        return np.inf
                    return safe_call(x)
                
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 200, 'maxfev': evals_per_restart,
                                            'xatol': 1e-8, 'fatol': 1e-10,
                                            'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 4: Final polish with tight tolerances
        if not check_deadline() and best_x is not None:
            remaining = budget - calls
            if remaining > 20:
                def objective_polish(x):
                    if check_deadline():
                        return np.inf
                    return safe_call(x)
                
                try:
                    res = minimize(objective_polish, best_x, method='Nelder-Mead',
                                   options={'maxiter': 200, 'maxfev': min(remaining, 100),
                                            'xatol': 1e-10, 'fatol': 1e-12,
                                            'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
            
            # If still have budget, do a few more random perturbations
            if not check_deadline() and best_x is not None:
                rng2 = np.random.default_rng(456)
                for _ in range(10):
                    if check_deadline():
                        break
                    perturbation = rng2.normal(0, 0.1, n)
                    x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                    f_try = safe_call(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
