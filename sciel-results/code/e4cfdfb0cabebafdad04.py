"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Random Search optimizer with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0
        time_exceeded = False

        def wrapped_func(x):
            nonlocal call_count, time_exceeded
            if call_count >= budget or time.time() > deadline:
                time_exceeded = True
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return fx
            except Exception:
                return np.inf

        # Structural Change: Adaptive Random Search (ARS)
        # Instead of local Nelder-Mead from random starts, we use a global
        # adaptive step-size random walk. This explores the space more broadly
        # and adapts to the local landscape curvature implicitly via step size.
        
        # 1. Initial random sampling to find a good starting point
        # Sample 10 points to initialize
        init_samples = 10
        init_budget = min(init_samples, budget // 10)
        
        for _ in range(init_budget):
            if call_count >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, d)
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            # Fallback if no calls were made
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # 2. Adaptive Random Search
        current_x = best_x.copy()
        current_f = best_f
        step_size = 2.0
        no_improve_count = 0
        max_no_improve = 15
        
        while call_count < budget and time.time() < deadline:
            # Propose new point
            direction = rng.standard_normal(d)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            new_x = current_x + step_size * direction
            new_x = np.clip(new_x, -5, 5)
            
            new_f = wrapped_func(new_x)
            
            if new_f < current_f:
                current_x = new_x
                current_f = new_f
                no_improve_count = 0
                step_size *= 1.05 # Increase step size slightly
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
            else:
                no_improve_count += 1
                step_size *= 0.9 # Decrease step size
                
                if no_improve_count > max_no_improve:
                    # Restart from best point with larger step to escape local optimum
                    current_x = best_x + rng.standard_normal(d) * 1.0
                    current_x = np.clip(current_x, -5, 5)
                    current_f = wrapped_func(current_x)
                    step_size = 3.0
                    no_improve_count = 0
                    
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()

        # 3. Final high-precision polish pass on the best point found
        if best_x is not None and not time_exceeded and call_count < budget:
            polish_budget = budget - call_count
            if polish_budget > 10:
                try:
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': polish_budget, 
                                            'xatol': 1e-6, 
                                            'fatol': 1e-6, 
                                            'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
