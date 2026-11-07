import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        rng = np.random.RandomState(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                return 1e10
            call_count += 1
            if call_count > self.budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def run_nm(x0, delta, max_iter):
            nonlocal call_count
            if call_count >= self.budget:
                return
            if time.time() > deadline - 10:
                return
                
            # Construct simplex
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            try:
                minimize(wrapped_func, x0, method='Nelder-Mead', 
                         options={'initial_simplex': simplex, 'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        # Phase 1: Exploration from origin
        run_nm(np.zeros(self.dim), 3.0, 30)
        
        # Phase 2: Refinement from current best
        run_nm(best_x.copy(), 0.5, 30)
        
        # Phase 3: Random restart exploration
        # Sample a random point in the box
        x_rand = rng.uniform(-4, 4, size=self.dim)
        run_nm(x_rand, 1.0, 30)
        
        # Phase 4: Polishing from the best found so far
        run_nm(best_x.copy(), 0.1, 50)
        
        return best_x, best_f
