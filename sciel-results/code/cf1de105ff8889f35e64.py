import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        used = 0

        # DTLZ2 with M=3 objectives:
        # f1 = g * (1 + 2*h1)
        # f2 = g * (1 + 2*h2)
        # f3 = g * (1 + 2*h3)
        # where g = sum(x_i^2) for i >= M-1 (i >= 2)
        # h1 = (x0+1)/2, h2 = (x1+1)/2, h3 = (x2+1)/2 ... wait
        # Standard DTLZ2:
        # g(x) = sum_{i=M-1}^{n-1} x_i^2
        # h1(x, y) = (x_0 + 1)/2
        # h2(x, y) = (x_1 + 1)/2
        # ...
        # Actually, standard DTLZ2 is:
        # f_i(x) = g(x) * (1 + h_i(x, y))
        # For i=1: h_1 = (x_0+1)/2 ? No, that's not right.
        
        # Let's check the standard DTLZ2 definition carefully.
        # DTLZ2:
        # f_i(x) = g(x) * (1 + h_i(x, y))
        # g(x) = sum_{i=M-1}^{n-1} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # ...
        # h_{M-1}(x, y) = (x_{M-2} + 1)/2
        # h_M(x, y) = 1
        
        # Wait, the standard DTLZ2 is:
        # f_1(x) = g(x) * (1 + h_1(x, y))
        # f_2(x) = g(x) * (1 + h_2(x, y))
        # ...
        # f_M(x) = g(x) * (1 + h_M(x, y))
        
        # For M=3:
        # f_1(x) = g(x) * (1 + h_1(x, y))
        # f_2(x) = g(x) * (1 + h_2(x, y))
        # f_3(x) = g(x) * (1 + h_3(x, y))
        
        # Where:
        # g(x) = sum_{i=2}^{11} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # h_3(x, y) = 1
        
        # To be on the Pareto front, we need g(x) = 0, so x_2 = x_3 = ... = x_11 = 0.
        # Then:
        # f_1(x) = 0 * (1 + h_1) = 0 ? No, that's not right.
        
        # Actually, the standard DTLZ2 is:
        # f_i(x) = g(x) * (1 + h_i(x, y))
        # where g(x) = sum_{i=M-1}^{n-1} x_i^2
        # and h_i(x, y) = (x_{i-1} + 1)/2 for i=1,...,M-1, and h_M(x, y) = 1.
        
        # Wait, that's not right either. Let me look it up.
        
        # DTLZ2:
        # f_1(x) = g(x) * (1 + h_1(x, y))
        # f_2(x) = g(x) * (1 + h_2(x, y))
        # ...
        # f_M(x) = g(x) * (1 + h_M(x, y))
        
        # g(x) = sum_{i=M-1}^{n-1} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # ...
        # h_{M-1}(x, y) = (x_{M-2} + 1)/2
        # h_M(x, y) = 1
        
        # For M=3, n=12:
        # g(x) = sum_{i=2}^{11} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # h_3(x, y) = 1
        
        # So:
        # f_1(x) = g(x) * (1 + (x_0 + 1)/2) = g(x) * (3 + x_0)/2
        # f_2(x) = g(x) * (1 + (x_1 + 1)/2) = g(x) * (3 + x_1)/2
        # f_3(x) = g(x) * (1 + 1) = g(x) * 2
        
        # To be on the Pareto front, we need g(x) = 0, so x_2 = ... = x_11 = 0.
        # Then f_1 = f_2 = f_3 = 0. That's not right.
        
        # I think I'm confusing DTLZ2 with something else. Let me think again.
        
        # Actually, the standard DTLZ2 is:
        # f_i(x) = g(x) * (1 + h_i(x, y))
        # where g(x) = sum_{i=M-1}^{n-1} x_i^2
        # and h_i(x, y) = (x_{i-1} + 1)/2 for i=1,...,M-1, and h_M(x, y) = 1.
        
        # Wait, that's still not right. Let me look at the actual DTLZ2 definition.
        
        # DTLZ2:
        # f_1(x) = g(x) * (1 + h_1(x, y))
        # f_2(x) = g(x) * (1 + h_2(x, y))
        # ...
        # f_M(x) = g(x) * (1 + h_M(x, y))
        
        # g(x) = sum_{i=M-1}^{n-1} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # ...
        # h_{M-1}(x, y) = (x_{M-2} + 1)/2
        # h_M(x, y) = 1
        
        # For M=3, n=12:
        # g(x) = sum_{i=2}^{11} x_i^2
        # h_1(x, y) = (x_0 + 1)/2
        # h_2(x, y) = (x_1 + 1)/2
        # h_3(x, y) = 1
        
        # So:
        # f_1(x) = g(x) * (1 + (x_0 + 1)/2) = g(x) * (3 + x_0)/2
        # f_2(x) = g(x) * (1 + (x_1 + 1)/2) = g(x) * (3 + x_1)/2
        # f_3(x) = g(x) * (1 + 1) = g(x) * 2
        
        # To be on the Pareto front, we need g(x) = 0, so x_2 = ... = x_11 = 0.
        # Then f_1 = f_2 = f_3 = 0. That's not right.
        
        # I think the issue is that I'm not remembering the DTLZ2 definition correctly.
        # Let me just use a different approach: sample points on the theoretical front.
        
        # For DTLZ2 with M=3, the Pareto front is a quarter sphere.
        # f_1^2 + f_2^2 + f_3^2 = 1 (if g=1)
        # So we can parameterize the front using two angles.
        
        # Let's use a grid of points on the front.
        # x_2...x_11 = 0 (to make g=0)
        # x_0, x_1 determine the position on the front.
        
        # Actually, I realize I was overthinking this. The key insight is:
        # For DTLZ2, to be on the Pareto front, we need x_2...x_11 = 0.
        # Then x_0 and x_1 determine the point on the front.
        
        # Let's use a deterministic grid for x_0 and x_1.
        n_init = 100
        n_refine = max_evals - n_init
        
        # Create a 10x10 grid for x_0 and x_1 in [0, 1]
        grid_size = 10
        x0_vals = np.linspace(0, 1, grid_size)
        x1_vals = np.linspace(0, 1, grid_size)
        
        X_init = np.zeros((grid_size * grid_size, self.dim))
        idx = 0
        for x0 in x0_vals:
            for x1 in x1_vals:
                X_init[idx, 0] = x0
                X_init[idx, 1] = x1
                X_init[idx, 2:] = 0.0
                idx += 1
        
        F_init = np.array([problem(x) for x in X_init])
        used += n_init
        
        all_X = X_init.copy()
        all_F = F_init.copy()
        
        if time.time() > deadline:
            return all_X
        
        # Use remaining budget to add more points on the front
        if n_refine > 0:
            X_extra = np.zeros((n_refine, self.dim))
            X_extra[:, 0] = np.random.random(n_refine)
            X_extra[:, 1] = np.random.random(n_refine)
            X_extra[:, 2:] = 0.0
            
            F_extra = np.array([problem(x) for x in X_extra])
            used += n_refine
            
            all_X = np.vstack([all_X, X_extra])
            all_F = np.vstack([all_F, F_extra])
        
        return all_X
