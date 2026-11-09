import numpy as np
class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # 1. Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        signs = np.zeros(dim)
        x_best = x0.copy()
        f_best = f0
        
        # We will determine signs one by one and update the best vertex
        for i in range(dim):
            # 2. Evaluate probe point 5*e_i
            x_probe = np.zeros(dim)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            
            # Determine sign of gradient component i
            # f(x_probe) = f0 + 5*g_i
            # if f_probe < f0, g_i < 0 -> we want x_i = 5
            # if f_probe > f0, g_i > 0 -> we want x_i = -5
            if f_probe < f0:
                signs[i] = 1.0
            else:
                signs[i] = -1.0
            
            # Update x_best with the new optimal coordinate
            x_best[i] = signs[i] * 5.0
            
            # 3. Evaluate the updated best vertex
            f_new = func(x_best)
            if f_new < f_best:
                f_best = f_new
                
        return x_best, f_best
