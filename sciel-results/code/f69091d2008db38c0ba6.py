"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-like adaptive step, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Initial evaluation
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            best_x = x.copy()
            best_f = fx
            
            sigma = 2.0
            no_improve = 0
            max_no_improve = 40
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.08, 8.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.92, 0.005)
                    no_improve += 1
                    
                if no_improve >= max_no_improve:
                    # Try multiple restart candidates with diverse strategies
                    candidates = []
                    # Perturb best_x with varying scales
                    for scale in [0.3, 0.7, 1.2, 2.0, 3.5]:
                        c = best_x + rng.normal(0, scale, dim)
                        c = np.clip(c, -5, 5)
                        candidates.append(c)
                    # Try random points in the box
                    for _ in range(3):
                        candidates.append(rng.uniform(-5, 5, dim))
                    # Try points near the center
                    for _ in range(2):
                        candidates.append(rng.normal(0, 1.0, dim))
                    
                    best_cand = None
                    best_cand_f = np.inf
                    for c in candidates:
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        cf = func(c)
                        calls_made += 1
                        if cf < best_cand_f:
                            best_cand_f = cf
                            best_cand = c
                    
                    if best_cand is not None:
                        x = best_cand
                        fx = best_cand_f
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        x = best_x.copy()
                        fx = best_f
                    
                    sigma = 2.0
                    no_improve = 0
                    
            # Final polish: multi-scale local search around best_x
            if calls_made < budget and time.time() < deadline:
                polish_scales = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
                for ps in polish_scales:
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    no_improve_polish = 0
                    max_no_improve_polish = 15
                    current_x = best_x.copy()
                    current_f = best_f
                    while calls_made < budget and time.time() < deadline:
                        y = np.clip(current_x + ps * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < current_f:
                            current_x = y
                            current_f = fy
                            no_improve_polish = 0
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                        else:
                            no_improve_polish += 1
                        if no_improve_polish >= max_no_improve_polish:
                            break
                        
        except Exception:
            pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
