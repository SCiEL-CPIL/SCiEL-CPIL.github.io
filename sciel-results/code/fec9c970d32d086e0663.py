import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_fx = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # 1. Initialization Phase
        # Sample a few points to find a good starting location
        # Include origin and random points
        init_points = [np.zeros(n)]
        for _ in range(9):
            init_points.append(rng.uniform(-5, 5, n))
            
        for p in init_points:
            fx = safe_call(p)
            if fx is None:
                break
        
        if best_x is None:
            return np.zeros(n), np.inf

        # 2. Local Refinement Phase using Powell's Method
        # Powell's method is derivative-free and effective for narrow valleys (like Discus)
        # It adapts directions, which is crucial for f11.
        
        # We allocate most of the budget to this phase
        # Each iteration of Powell in n dims uses approx n function calls for the main loop
        # plus some for the conjugate direction update.
        # Let's estimate max iterations.
        # If we use 450 calls for optimization, and n=5, that's roughly 90 iterations.
        # SciPy's Powell 'maxiter' counts iterations of the main loop.
        
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            
            def powell_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls_made += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_clipped.copy()
                return fx

            try:
                # xatol and fatol control convergence
                # adaptive=True allows the simplex to adapt (though Powell doesn't use simplex in the same way, it's a valid option)
                # maxiter: limit iterations to prevent timeout
                max_iter = max(10, int(remaining_budget // (n + 2)))
                
                res = minimize(powell_func, best_x, method='Powell',
                               options={'maxiter': max_iter, 
                                        'xatol': 1e-6, 
                                        'fatol': 1e-6,
                                        'adaptive': True})
            except Exception:
                pass

        # 3. Final Polishing (Optional)
        # If we have significant budget left, run Nelder-Mead for a quick polish
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline:
            
            def nm_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls_made += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_clipped.copy()
                return fx

            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxiter': int(remaining_budget // (n + 1)), 
                                        'xatol': 1e-8, 
                                        'fatol': 1e-8,
                                        'adaptive': True})
            except Exception:
                pass

        return best_x, best_fx
