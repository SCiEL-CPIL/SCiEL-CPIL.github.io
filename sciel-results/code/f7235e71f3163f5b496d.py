import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Halton sequence for space-filling coverage
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        def halton_sequence(n, dim, bases=None):
            if bases is None:
                # First n primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                bases = primes[:dim]
            points = []
            for i in range(1, n + 1):
                point = [halton(i, b) for b in bases]
                points.append(point)
            return np.array(points)
        
        # Use Halton sequence for initial exploration
        n_halton = min(40, budget)
        halton_pts = halton_sequence(n_halton, dim)
        # Map from [0,1]^d to [-5,5]^d
        halton_pts = halton_pts * 10.0 - 5.0
        
        for x in halton_pts:
            safe_call(x)
            if calls_made >= budget or time.time() > deadline:
                break
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Multi-start Nelder-Mead on best points
        # Evaluate all points and sort
        all_points = []
        for x in halton_pts:
            if calls_made >= budget or time.time() > deadline:
                break
            # We already evaluated these, but let's track them
            # Actually, let's just use the best ones we found
        
        # Get best x from safe_call tracking
        # We need to track evaluated points
        # Let me restructure to track points
        
        # Actually, let's just run NM from best_x and a few other good points
        # Use NM from the best point found so far
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        # Run Nelder-Mead from best_x
        if best_x is not None:
            def nm_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return best_f
                return safe_call(x)
            
            try:
                res = minimize(
                    nm_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # If we still have budget, try a few more NM runs from diverse points
        if calls_made < budget and time.time() < deadline:
            # Generate a few diverse starting points
            rng = np.random.default_rng(42)
            for _ in range(3):
                if calls_made >= budget or time.time() > deadline:
                    break
                x0 = rng.uniform(-5, 5, size=dim)
                def nm_func2(x):
                    if calls_made >= budget or time.time() > deadline:
                        return best_f
                    return safe_call(x)
                
                try:
                    res = minimize(
                        nm_func2,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 300,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass
        
        return best_x, best_f
