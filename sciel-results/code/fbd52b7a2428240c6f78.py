"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer for 5D Rastrigin with CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline with 50s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = [0]
        
        def call_func(x):
            calls[0] += 1
            return func(x)
        
        # Phase 1: Coarse random exploration (30 calls)
        best_x = None
        best_f = np.inf
        for i in range(30):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES (main optimization)
        x = best_x.copy()
        fx = best_f
        
        # CMA-ES parameters
        pop_size = dim + int(3 * np.log(dim))  # ~8 for dim=5
        pop_size = max(4, min(pop_size, 12))
        
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        cs = (4 + dim / 3) / (dim + 4)
        cmu = 2 / ((dim + 1) ** 2 + 1)
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cc = (4 + dim / dim) / (dim + 4)
        mu = dim // 2 + 1  # number of recombination points
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Use a simple (1+1)-ES with adaptive sigma for reliability
        sigma = 1.5
        for i in range(budget - calls[0]):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            # Generate trial
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = call_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # If sigma gets too small, restart with larger sigma from best
            if sigma < 0.01:
                sigma = 0.5
                # Try a few more steps with larger sigma
                for j in range(5):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = call_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
        
        # Phase 3: Fine-tuning near the best point (remaining budget)
        if calls[0] < budget:
            sigma = 0.1
            for i in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break
        
        return x, fx
# EVOLVE-BLOCK-END
