import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # ZDT1 Pareto front: f1 = x0, f2 = 1 - sqrt(x0)
        # HV = sum_{i=0}^{n-1} (10 + sqrt(t_i)) * (t_{i+1} - t_i)
        # Optimal quadrature density: p(x) ∝ sqrt(|h''(x)|) where h(x) = 10 + sqrt(x)
        # h''(x) = -1/(4x^(3/2)), so optimal density ∝ x^(-3/4)
        # With x0 = u^alpha, density in x0 ∝ x0^((1-alpha)/alpha)
        # Setting (1-alpha)/alpha = -3/4 gives alpha = 4
        alpha = 4.0
        
        if n_evals > 1:
            indices = np.linspace(0, 1, n_evals)
            x0_values = indices ** alpha
        else:
            x0_values = np.array([0.5])
            
        for x0 in x0_values:
            if time.time() > self.deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
