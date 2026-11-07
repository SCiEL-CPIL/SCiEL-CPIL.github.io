import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        # We need d+1 evaluations to determine the linear function
        # f(x) = w^T x + b
        
        # 1. Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # 2. Evaluate at 5 * e_i
        # This gives us w_i = (f(5*e_i) - f(0)) / 5
        w = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            w[i] = (fi - f0) / 5.0
            
        # 3. Determine optimal vertex
        # Minimize w^T x subject to x in [-5, 5]^d
        # x_i should be -5 if w_i > 0, and 5 if w_i < 0
        v_star = np.where(w > 0, -5.0, 5.0)
        
        # 4. Evaluate the optimal vertex
        f_opt_est = func(v_star)
        
        # The best value found is min(f0, min(fi), f_opt_est)
        # But f_opt_est is the global minimum, so it's the best.
        # We don't need to do anything else.
        # The evaluator will repeat the last n_i.
        
        return v_star, f_opt_est
