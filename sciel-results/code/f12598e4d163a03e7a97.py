import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration with Nelder-Mead
        # Nelder-Mead is derivative-free and robust for Rosenbrock-like landscapes.
        # We use multiple starts.
        
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        
        # Axes points
        for i in range(self.dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(self.dim)
                p[i] = val
                starts.append(p)
        
        # Random points
        n_starts = 10
        for _ in range(n_starts):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
            
        rng.shuffle(starts)
        
        for i, x0 in enumerate(starts):
            if not time_ok() or call_count >= self.budget - 100: 
                break
            
            remaining_for_phase1 = self.budget - 100 - call_count
            if remaining_for_phase1 <= 0:
                break
            
            max_fun_this_start = min(30, remaining_for_phase1)
            
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': max_fun_this_start, 'adaptive': True}
                )
            except Exception:
                pass
            
            if not time_ok():
                break

        # Phase 2: Intensive Refinement with L-BFGS-B
        # L-BFGS-B is very fast for smooth functions once close to the minimum.
        if time_ok() and call_count < self.budget - 5:
             try:
                 res = minimize(
                     tracked_func,
                     best_x,
                     method='L-BFGS-B',
                     bounds=[(-5.0, 5.0)] * self.dim,
                     options={'maxiter': 500, 'maxfun': self.budget - call_count - 10, 'ftol': 1e-12, 'gtol': 1e-8}
                 )
             except Exception:
                pass
        
        # Phase 3: Final Polish (1+1-ES)
        if time_ok() and call_count < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            remaining = self.budget - call_count
            
            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = tracked_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-6:
                    break

        return best_x, best_fx
