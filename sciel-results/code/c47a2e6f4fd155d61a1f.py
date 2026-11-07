import numpy as np
import scipy.optimize
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            # Clip to bounds
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Stratified Sampling
        # 2^dim intervals
        n_intervals = 2
        # Generate corners of the sub-boxes
        # Actually, just picking random points in each of the 2^dim hyper-rectangles is sufficient.
        
        # Create list of points
        points = []
        # Bitwise combinations for 5 dims
        for i in range(2 ** self.dim):
            # Determine which sub-box
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bit = (i >> d) & 1
                if bit == 0:
                    # [-5, 0]
                    x[d] = np.random.uniform(-5, 0)
                else:
                    # [0, 5]
                    x[d] = np.random.uniform(0, 5)
            points.append(x)
            
        # Evaluate all
        for p in points:
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            wrapped_func(p)
            
        if self.best_x is None:
             # Fallback if no calls made? Should not happen
             return np.zeros(self.dim), 0.0

        # Phase 2: Local Optimization
        # Use Powell method
        # maxiter: 500 calls / 5 dims = 100 iterations approx
        try:
            # x0 is the best found so far
            res = scipy.optimize.minimize(
                wrapped_func, 
                self.best_x, 
                method='Powell',
                options={
                    'maxiter': 1000, # Let it run until budget/time
                    'ftol': 1e-12,
                    'xtol': 1e-12
                }
            )
            if res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = res.x
        except TimeoutError:
            pass
        except Exception as e:
            # Catch other errors and return best so far
            pass
        
        # Phase 3: High-precision polish with Nelder-Mead if budget remains
        if self.call_count < self.budget - 10 and time.time() < self.deadline - 10:
            try:
                res2 = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res2.fun < self.best_f:
                    self.best_f = res2.fun
                    self.best_x = res2.x
            except TimeoutError:
                pass
            except Exception as e:
                pass

        return self.best_x, self.best_f
