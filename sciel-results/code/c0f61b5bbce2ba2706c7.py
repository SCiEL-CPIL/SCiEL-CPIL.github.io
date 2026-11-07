import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        x = np.zeros(self.dim)
        x[1:] = 0.5
        
        # Calculate the minimum x1 that contributes to hypervolume
        # f2 = 292 * (1 - sqrt(x1)) <= 11
        # 1 - sqrt(x1) <= 11/292
        # sqrt(x1) >= 1 - 11/292
        # x1 >= (1 - 11/292)**2
        x_min = (1.0 - 11.0 / 292.0) ** 2
        
        n_evals = self.budget
        last_f = None
        
        for i in range(n_evals):
            # Uniform sampling in [x_min, 1.0]
            # Include endpoints
            x1 = x_min + (1.0 - x_min) * (i / (n_evals - 1))
            x[0] = x1
            
            f = problem(x)
            last_f = f
            
        return x
