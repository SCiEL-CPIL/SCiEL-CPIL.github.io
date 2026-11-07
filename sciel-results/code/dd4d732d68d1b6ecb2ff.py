"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with step-norm adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.0
        calls_made = 1
        budget = self.budget
        dim = self.dim
        
        # Phase 1: (1+1)-ES with step-norm-based sigma adaptation
        # Reserve last 50 calls for Nelder-Mead polish
        es_budget = budget - 50
        for i in range(es_budget - 1):
            if time.time() > deadline - 5:
                break
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1
            step_norm = np.linalg.norm(y - x)
            if fy < fx:
                x, fx = y, fy
                # Adapt sigma based on step norm relative to expected
                # Expected step norm for standard normal in dim dimensions
                expected_norm = np.sqrt(dim)
                ratio = step_norm / (sigma * expected_norm)
                # Conservative adaptation
                sigma *= np.exp(0.3 * (ratio - 1.0))
                sigma = np.clip(sigma, 1e-6, 10.0)
            else:
                sigma *= 0.95
                sigma = np.clip(sigma, 1e-6, 10.0)
        
        # Phase 2: Nelder-Mead polish with remaining budget
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 5:
            from scipy.optimize import minimize
            try:
                result = minimize(func, x, method='Nelder-Mead', 
                                  options={'maxiter': remaining, 'adaptive': True,
                                           'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < fx:
                    x, fx = result.x, result.fun
            except Exception:
                pass
        
        return x, fx
# EVOLVE-BLOCK-END
