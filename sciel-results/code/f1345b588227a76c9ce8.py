import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_fx = np.inf
        
        def latin_hypercube_sample(n, d, low=-5.0, high=5.0):
            points = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    points[i, j] = low + (high - low) * (i + np.random.uniform()) / n
            return points

        # Wrapper to count calls
        def func_wrapper(x):
            nonlocal n_calls
            n_calls += 1
            return func(x)

        # Use Latin Hypercube for better initial coverage
        num_init = 20
        init_x = latin_hypercube_sample(num_init, dim)
        init_f = np.array([func_wrapper(x) for x in init_x])
        n_calls = num_init # Already incremented by wrapper, but let's be explicit if we use wrapper for init
        
        idx = np.argmin(init_f)
        x = init_x[idx].copy()
        best_fx = init_f[idx]
        best_x = x.copy()
        
        if time.time() > deadline - 10:
             try:
                 res = minimize(func_wrapper, x, method='Nelder-Mead', options={'maxfev': budget - n_calls, 'maxiter': budget - n_calls})
                 if res.fun < best_fx:
                     best_fx = res.fun
                     best_x = res.x
             except:
                 pass
             return best_x, best_fx

        # Coordinate Descent
        while n_calls < budget:
            if time.time() > deadline - 5:
                break
            
            improved = False
            prev_best_fx = best_fx
            
            for i in range(dim):
                if n_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                    
                # Define 1D function for axis i
                def f_1d(t, x=x, i=i):
                    xi = x.copy()
                    xi[i] = t
                    return func_wrapper(xi)
                
                try:
                    # Use bounded Brent's method
                    res_1d = minimize_scalar(f_1d, bounds=(-5.0, 5.0), method='bounded', options={'maxiter': 30, 'xatol': 1e-8})
                    
                    if res_1d.fun < best_fx:
                        best_fx = res_1d.fun
                        x[i] = res_1d.x
                        best_x = x.copy()
                        improved = True
                        
                except Exception:
                    pass
            
            if not improved:
                # If no improvement in a full pass, we might be stuck.
                # Try a small random perturbation or break.
                # Given the directive, we stick to CD. Let's break to avoid wasting budget.
                # Or we can try a few random restarts if budget allows.
                # Let's break for now, as CD is the requested structural change.
                break
        
        # Polish with Nelder-Mead if budget remains
        remaining = budget - n_calls
        if remaining > 15:
            try:
                res = minimize(func_wrapper, best_x, method='Nelder-Mead', options={'maxfev': remaining, 'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
