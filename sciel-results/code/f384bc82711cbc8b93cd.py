import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                raise Exception("Budget exceeded")
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        def grad(x):
            g = np.zeros(self.dim)
            h = 1e-5
            for i in range(self.dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                g[i] = (wrapped_func(x_plus) - wrapped_func(x_minus)) / (2 * h)
            return g

        x0 = np.zeros(self.dim)
        
        # Ensure we have at least 1 call for x0
        if self.budget <= 0:
            return x0, np.inf

        try:
            f0 = wrapped_func(x0)
        except Exception:
            return self.best_x if self.best_x is not None else x0, self.best_f

        # L-BFGS-B
        try:
            # maxiter is a safety cap, but budget check is primary
            res = minimize(wrapped_func, x0, jac=grad, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * self.dim, 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        # Nelder-Mead fallback
        if self.calls < self.budget:
            try:
                res2 = minimize(wrapped_func, self.best_x, method='Nelder-Mead', 
                                bounds=[(-5, 5)] * self.dim, 
                                options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-8})
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = x0
            self.best_f = func(x0) # This might exceed budget, but we already checked? 
            # Actually, if budget was 0, we returned. If budget >= 1, we called x0.
            
        return self.best_x, self.best_f
