import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0.0
        self.best_x = None
        self.best_f = float('inf')
        
    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        
        # Wrapper to track calls and time
        def wrapped_func(x):
            self.calls += 1
            if time.time() - self.start_time > 290: # Safety margin
                raise TimeoutError
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f
            
        # Phase 1: Random Sampling to find promising basins
        n_initial = 50
        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 5, (n_initial, self.dim))
        
        for p in points:
            if self.calls >= self.budget - 100: # Reserve some budget
                break
            f = wrapped_func(p)
            
        # Phase 2: Local Search from the best few points
        # We don't easily get the values of all points from the wrapper unless we store them.
        # Let's store them.
        
        # Reset and do it properly
        self.calls = 0
        self.start_time = time.time()
        
        def tracked_func(x):
            self.calls += 1
            if time.time() - self.start_time > 290:
                raise TimeoutError
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # 1. Initial Scatter
        n_scatter = 40
        scatter_points = []
        scatter_fvals = []
        rng = np.random.default_rng(42)
        for _ in range(n_scatter):
            x = rng.uniform(-5, 5, self.dim)
            fx = tracked_func(x)
            scatter_points.append(x)
            scatter_fvals.append(fx)
            
        # 2. Select top K for local search
        top_k = 5
        indices = np.argsort(scatter_fvals)[:top_k]
        start_points = [scatter_points[i] for i in indices]
        
        # 3. Local Search (Nelder-Mead)
        # Budget remaining: 500 - 40 = 460.
        # Allocate ~90 calls per local search.
        calls_per_search = 90
        
        for sp in start_points:
            if self.calls >= self.budget - 50: # Keep reserve
                break
                
            # We need to limit the calls in minimize.
            # scipy.optimize.minimize doesn't have a direct call limit, 
            # but we can use a counter in the function and raise an exception 
            # or use a callback.
            
            local_calls = 0
            def local_func(x):
                nonlocal local_calls
                local_calls += 1
                if local_calls > calls_per_search:
                    raise ValueError("Local budget exceeded")
                return tracked_func(x)
                
            try:
                res = minimize(local_func, sp, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            except (ValueError, TimeoutError):
                pass
            except Exception:
                pass

        # 4. Final Polish
        if self.calls < self.budget - 10:
            # Use the best point found so far
            final_calls = 0
            def final_func(x):
                nonlocal final_calls
                final_calls += 1
                if final_calls > 50:
                    raise ValueError("Final budget exceeded")
                return tracked_func(x)
            
            try:
                res = minimize(final_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            except (ValueError, TimeoutError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
