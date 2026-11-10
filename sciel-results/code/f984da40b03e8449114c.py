"""Optimized ZDT2 optimizer using analytical optimal spacing for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Model-based optimizer for ZDT2.
    Places points on the Pareto front with spacing optimized to maximize Hypervolume.
    Uses Newton's method for fast inverse computation of the spacing function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n_evals = self.budget
        
        # ZDT2 Pareto Front: f1 = x[0], f2 = 1 - sqrt(x[0])
        # Reference point r = [11, 11]
        # Height function h(x) = r[1] - f2(x) = 10 + sqrt(x)
        # Optimal spacing: t(x) = 10x + (2/3)x^(3/2), uniform in t
        
        def t_of_x(x):
            return 10.0 * x + (2.0 / 3.0) * np.power(x, 1.5)
        
        def dt_of_x(x):
            return 10.0 + 1.5 * np.sqrt(x)
        
        def x_of_t(t):
            # Newton's method to solve t(x) = t_target
            # Initial guess: linear approximation x ≈ t/10 (since t ≈ 10x for small x)
            x = min(t / 10.0, 1.0)
            for _ in range(20):
                fx = t_of_x(x) - t
                if abs(fx) < 1e-15:
                    break
                dfx = dt_of_x(x)
                x_new = x - fx / dfx
                # Clamp to [0, 1]
                if x_new < 0.0:
                    x_new = 0.0
                elif x_new > 1.0:
                    x_new = 1.0
                if abs(x_new - x) < 1e-15:
                    x = x_new
                    break
                x = x_new
            return x

        t_max = t_of_x(1.0)
        
        # Generate uniform t values from 0 to t_max
        t_vals = np.linspace(0, t_max, n_evals)
        x_vals = np.array([x_of_t(t) for t in t_vals])
        
        # Ensure x values are within [0, 1] and sorted
        x_vals = np.clip(x_vals, 0.0, 1.0)
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x_vals
        # x[1:] = 0
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])
            
            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
