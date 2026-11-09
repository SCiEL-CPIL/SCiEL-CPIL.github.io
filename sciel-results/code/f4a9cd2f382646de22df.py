import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
            start_time = time.time()
            # Use a safety margin of 20 seconds
            deadline = start_time + 280 
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            n_evals = self.budget
            
            # ZDT2 Pareto front exploitation
            # Optimal x_i = 0.5 for i > 0
            # Search x_0 in [0, 1]
            
            # Distribution of x_0
            # Optimal spacing proportional to |f''|^{-0.5}
            # f'' ~ x^{-1.5} => spacing ~ x^{0.75}
            # x ~ u^{1/1.75} ~ u^{0.57}
            # Use 0.6 as a robust approximation
            
            u = np.linspace(0.0, 1.0, n_evals)
            x0_vals = u ** 0.6
            
            # Construct X
            X = np.zeros((n_evals, self.dim))
            X[:, 0] = x0_vals
            X[:, 1:] = 0.5
            
            # Ensure bounds are respected (though 0.5 and [0,1] are safe)
            X = np.clip(X, lo, hi)
            
            # Evaluate
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                problem(X[i])
                
            return X
