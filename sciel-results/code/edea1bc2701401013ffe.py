"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = np.random.uniform(-5, 5, dim)
        best_fx = func(best_x)
        n_calls += 1
        
        def time_left():
            return deadline - time.time()
            
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling for better initial coverage
        # Use 40 LHS points to spread out across the domain
        n_lhs = 40
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + 0.5) / n_lhs * 10.0 - 5.0
        
        for x in lhs_points:
            if n_calls >= budget or time_left() < 10:
                break
            fx = func(x)
            n_calls += 1
            update_best(x, fx)

        # Phase 2: CMA-ES-like adaptive search using scipy minimize with COBYLA
        # COBYLA handles the box constraints and doesn't require gradients
        # It's robust for the step ellipsoidal landscape
        
        remaining = budget - n_calls
        if remaining > 50 and best_x is not None:
            try:
                # Use COBYLA which respects bounds without penalty
                res = minimize(func, best_x, method='COBYLA',
                               constraints=[{'type': 'ineq', 'fun': lambda x, i=i: x[i] + 5} for i in range(dim)] +
                                       [{'type': 'ineq', 'fun': lambda x, i=i: 5 - x[i]} for i in range(dim)],
                               options={'maxiter': min(remaining // 2, 200), 'rhobeg': 1.0, 'catol': 1e-8})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Coordinate descent refinement with adaptive step sizes
        # This exploits the structure of the step ellipsoidal function
        for p in range(30):
            if n_calls >= budget or time_left() < 5:
                break
            
            improved = False
            for d in range(dim):
                if n_calls >= budget or time_left() < 5:
                    break
                
                # Try a few step sizes in both directions
                step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                best_local_fx = best_fx
                best_local_x = best_x.copy()
                
                for s in step_sizes:
                    for direction in [1, -1]:
                        if n_calls >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[d] += direction * s
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = func(x_try)
                        n_calls += 1
                        if fx_try < best_local_fx:
                            best_local_fx = fx_try
                            best_local_x = x_try.copy()
                
                if best_local_fx < best_fx - 1e-10:
                    best_x = best_local_x.copy()
                    best_fx = best_local_fx
                    improved = True
            
            if not improved:
                break

        # Phase 4: Final fine-tuning with small perturbations
        remaining = budget - n_calls
        if remaining > 20 and best_x is not None:
            for _ in range(min(remaining, 50)):
                if n_calls >= budget or time_left() < 2:
                    break
                # Small random perturbation
                x_try = best_x + np.random.normal(0, 0.01, dim)
                x_try = np.clip(x_try, -5, 5)
                fx_try = func(x_try)
                n_calls += 1
                update_best(x_try, fx_try)

        return best_x, best_fx
# EVOLVE-BLOCK-END
