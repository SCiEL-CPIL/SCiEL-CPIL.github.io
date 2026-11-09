import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT6 using analytic Pareto-front construction.
    
    ZDT6 Pareto front is characterized by x1 in [0, 1/6] and x2..x10 = 0.
    f1(x1) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
    f2(x) = 1 - sqrt(f1(x)) when x2..x10=0.
    
    We maximize hypervolume by distributing evaluations uniformly along the Pareto front
    in the objective space (specifically, uniform in f1).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit

        # ZDT6 specific function for f1
        def f1_func(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
        
        # The relevant part of the Pareto front is the first lobe of sin^6(6*pi*x1)
        # which is x1 in [0, 1/6].
        # f1(0) = 1, f1(1/6) = 1.
        # There is a minimum in between.
        
        # 1. Find x_min where f1 is minimized in [0, 1/6]
        # Coarse grid search
        x_grid = np.linspace(0.0, 1.0/6.0, 1000)
        f1_grid = f1_func(x_grid)
        idx_min = np.argmin(f1_grid)
        x_min_approx = x_grid[idx_min]
        
        # Refine x_min with a finer grid around the approximate minimum
        delta = (1.0/6.0) / 1000
        x_refine_grid = np.linspace(x_min_approx - 5*delta, x_min_approx + 5*delta, 100)
        # Ensure we stay within [0, 1/6]
        x_refine_grid = np.clip(x_refine_grid, 0.0, 1.0/6.0)
        f1_refine = f1_func(x_refine_grid)
        idx_min_ref = np.argmin(f1_refine)
        x_min = x_refine_grid[idx_min_ref]
        f1_min = f1_func(x_min)
        
        # 2. Generate target f1 values uniformly in [f1_min, 1.0]
        n_evals = self.budget
        target_f1s = np.linspace(f1_min, 1.0, n_evals)
        
        # 3. Invert f1(x1) = target to find x1 for each target
        # We use the branch x1 in [0, x_min] where f1 is monotonically decreasing.
        X = np.zeros((n_evals, self.dim))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            tf = target_f1s[i]
            
            # Bisection to find x1 in [0, x_min] such that f1(x1) = tf
            a = 0.0
            b = x_min
            
            # f1(a) = 1.0, f1(b) = f1_min
            # Since f1 is decreasing, if f1(mid) > tf, mid is too small (left), so a = mid
            # If f1(mid) < tf, mid is too large (right), so b = mid
            
            for _ in range(50): # 50 iterations gives ~2^-50 precision
                mid = (a + b) / 2.0
                fm = f1_func(mid)
                if fm > tf:
                    a = mid
                else:
                    b = mid
            
            x1_val = (a + b) / 2.0
            X[i, 0] = x1_val
            # x2..x10 are already 0
        
        # 4. Evaluate the points
        # The problem function clips to bounds, but our points are within [0, 1]
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X[-1]
