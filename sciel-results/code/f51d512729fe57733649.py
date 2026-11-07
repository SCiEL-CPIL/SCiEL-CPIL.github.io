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
        deadline = start_time + 275.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        try:
            # Phase 1: Multi-start exploration (first 100 calls)
            n_starts = min(10, budget // 50)
            for i in range(n_starts):
                if calls_made >= budget or time.time() >= deadline:
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                fx = evaluate(x)
            
            # Phase 2: (1+1)-ES with adaptive sigma and restarts
            x = best_x.copy()
            fx = best_f
            sigma = 1.5
            no_improve = 0
            max_no_improve = 30
            restart_count = 0
            max_restarts = 5
            
            while calls_made < budget - 50 and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.06, 6.0)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.94, 0.01)
                    no_improve += 1
                    
                if no_improve >= max_no_improve and restart_count < max_restarts:
                    # Restart with larger perturbation from best
                    restart_count += 1
                    x = best_x + rng.normal(0, 2.0 * (1 + restart_count * 0.5), dim)
                    x = np.clip(x, -5, 5)
                    fx = evaluate(x)
                    sigma = 1.0 + restart_count * 0.5
                    no_improve = 0
            
            # Phase 3: Intensive local polish (last ~50 calls)
            if calls_made < budget and time.time() < deadline:
                px = best_x.copy()
                pfx = best_f
                polish_sigma = 0.1
                no_polish_improve = 0
                max_polish_no_improve = 15
                
                while calls_made < budget and time.time() < deadline:
                    py = np.clip(px + polish_sigma * rng.standard_normal(dim), -5, 5)
                    pfy = evaluate(py)
                    
                    if pfy < pfx:
                        px = py
                        pfx = pfy
                        polish_sigma = min(polish_sigma * 0.95, 0.005)
                        no_polish_improve = 0
                    else:
                        polish_sigma = max(polish_sigma * 0.85, 0.0005)
                        no_polish_improve += 1
                    
                    if no_polish_improve >= max_polish_no_improve:
                        # Try a fresh restart for polish
                        if calls_made < budget - 5:
                            px = best_x + rng.normal(0, 0.1, dim)
                            px = np.clip(px, -5, 5)
                            pfx = evaluate(px)
                            polish_sigma = 0.05
                            no_polish_improve = 0
                        else:
                            break

        except Exception:
            pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
