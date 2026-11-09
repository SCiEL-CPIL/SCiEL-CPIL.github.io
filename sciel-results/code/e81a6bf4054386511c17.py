import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(42)
        calls_used = 0
        x_best = np.zeros(self.dim)
        f_best = np.inf
        
        # Phase 1: Systematic Exploration with Halton Sequence
        # Halton sequences provide better space-filling properties than random points,
        # which helps cover the domain more evenly and find the global basin early.
        n_global = 30
        halton_points = self._halton_sequence(n_global, self.dim)
        
        # Map [0, 1]^d to [-5, 5]^d
        points = halton_points * 10 - 5
        
        for i in range(n_global):
            if calls_used >= self.budget: break
            if time.time() > deadline: break
            
            p = points[i]
            fp = func(p)
            calls_used += 1
            
            if fp < f_best:
                f_best = fp
                x_best = p.copy()
        
        # Phase 2: Local Refinement with (1+1)-ES on the best point
        # Use remaining budget for local search to converge to the minimum.
        remaining = self.budget - calls_used
        if remaining > 0:
            sigma = 0.5
            x_curr = x_best.copy()
            f_curr = f_best
            
            for _ in range(remaining):
                if time.time() > deadline: break
                
                y = np.clip(x_curr + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    sigma *= 1.1
                    if f_curr < f_best:
                        x_best, f_best = x_curr, f_curr
                else:
                    sigma *= 0.9
                    
        return x_best, f_best
