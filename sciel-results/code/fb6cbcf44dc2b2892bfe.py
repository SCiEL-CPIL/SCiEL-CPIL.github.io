import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start Nelder-Mead + L-BFGS-B Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        rng = np.random.default_rng(42)
        
        # Priority-ordered starts
        starts = [np.ones(self.dim)]
        
        for scale in [0.5, 1.0, 2.0]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        
        for _ in range(5):
            starts.append(rng.uniform(-3, 5, self.dim))
        
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        bounds = [(-5, 5)] * self.dim

        # Phase 1: Nelder-Mead (Derivative-Free)
        # Nelder-Mead is robust and doesn't require gradient estimation, saving calls.
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 20:
                continue
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: L-BFGS-B Polish
        # Use the best point found by Nelder-Mead as the starting point for L-BFGS-B.
        # L-BFGS-B is very fast locally.
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls_made
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'ftol': 1e-12,
                            'gtol': 1e-10
                        }
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass
        
        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
