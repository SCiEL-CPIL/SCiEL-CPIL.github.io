import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 300s limit, keep 20s margin
        
        # Wrapper to track calls and best value
        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            self.calls += 1
            # Ensure x is a valid numpy array of the correct shape
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                x = x[:self.dim]
            
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # Multi-Start Powell
        rng = np.random.default_rng(42)
        
        # Number of restarts
        # With 500 budget and dim 5, Powell is efficient.
        # Let's try 5 restarts.
        n_restarts = 5
        
        # Distribute budget roughly equally, but allow for early termination
        # Each Powell iteration uses ~2*dim calls.
        # Let's just let scipy run until it uses up the allocated slice or converges.
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Random start in [-4, 4]^dim
            x0 = rng.uniform(-4, 4, self.dim)
            
            # Determine maxiter based on remaining budget
            remaining = self.budget - self.calls
            # Powell iteration cost is variable, roughly 2*dim to 5*dim.
            # Be conservative: assume 3*dim calls per iter.
            max_iter = max(1, remaining // (3 * self.dim))
            
            try:
                # Use Powell method
                result = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-8}
                )
            except (StopIteration, TimeoutError):
                break
            except Exception:
                # If scipy fails, try Nelder-Mead or just continue
                # Fallback to Nelder-Mead for this restart
                try:
                    result = scipy.optimize.minimize(
                        wrapped_func, 
                        x0, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining // 2, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                except (StopIteration, TimeoutError):
                    break
                except Exception:
                    continue

        if self.best_x is None:
            # Should not happen, but safe fallback
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)

        return self.best_x, self.best_f
