import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_scan(self, func, n, dim, deadline, calls_left):
        """Perform a Sobol scan to find a good starting point."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n)
        except Exception:
            # Fallback to Halton or random
            rng = np.random.default_rng(42)
            u = rng.random((n, dim))
        
        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)
        
        best_val = float('inf')
        best_x = None
        
        for i in range(n):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            x = pts[i]
            fx = func(x)
            calls_left[0] -= 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
        
        return best_x, best_val

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            return float(func(x))

        rng = np.random.default_rng(12345)

        # Phase 1: Sobol Scan
        n_scan = min(50, budget // 5)
        if n_scan < 10:
            n_scan = 10
            
        best_x, best_fx = self._sobol_scan(guarded, n_scan, dim, deadline, calls_left)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = guarded(best_x)

        # Phase 2: (1+1)-ES from best_x
        x = best_x
        fx = best_fx
        sigma = 1.0 # Initial step size
        
        # Adapt sigma based on success
        # 1/5 rule: if success rate > 1/5, increase sigma, else decrease
        # Simple version: increase on success, decrease on failure
        # For Bent Cigar, we want to keep sigma large enough to move along the flat axis
        # but small enough to converge on the steep axis.
        
        for _ in range(calls_left[0]):
            if time.time() >= deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_left[0] -= 1
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma to explore further
                # 1.5 is aggressive, 1.1 is conservative. 
                # Inspiration used 1.5. Let's try 1.2 for stability.
                sigma *= 1.2
            else:
                # Decrease sigma to refine
                sigma *= 0.85

        return x, fx
