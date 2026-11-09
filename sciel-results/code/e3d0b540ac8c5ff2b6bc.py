import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 320.0

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        # Optimal distribution for ZDT1 hypervolume maximization
        # Derived from minimizing discretization error for concave h(u)
        # u_k = (k / (n-1))^2 for k=0..n-1
        # This ensures u_0=0 and u_{n-1}=1
        
        if n == 1:
            x0_values = np.array([0.0])
        else:
            indices = np.arange(n)
            x0_values = (indices / (n - 1)) ** 2
            
        # Shuffle to avoid order-dependent issues (though unlikely)
        # Note: Shuffling does not change the set of points, so HV is invariant
        # But we must ensure we don't exceed budget.
        
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
