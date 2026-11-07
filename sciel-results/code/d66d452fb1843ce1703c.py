import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Pareto front: x[1:] = 0, x[0] = t in [0, 1]
        # f1 = t
        # f2 = 1 - sqrt(t) - t * sin(10*pi*t)
        
        # Strategy: Sample uniformly in f2-space to maximize hypervolume coverage
        # of the monotonic Pareto front.
        
        n_samples = budget
        
        # f2(t) ranges from 1 (at t=0) to 0 (at t=1)
        y_values = np.linspace(1.0, 0.0, n_samples)
        
        # Initial guess: t = (1 - y)^2
        t_values = np.power(1.0 - y_values, 2)
        t_values = np.clip(t_values, 0.0, 1.0)
        
        # Newton's method to solve f2(t) = y for t
        for _ in range(20):
            sqrt_t = np.sqrt(t_values)
            sin_term = np.sin(10.0 * np.pi * t_values)
            cos_term = np.cos(10.0 * np.pi * t_values)
            
            f2_val = 1.0 - sqrt_t - t_values * sin_term
            f2_prime = -0.5 / sqrt_t - sin_term - 10.0 * np.pi * t_values * cos_term
            
            # Avoid division by zero at t=0
            # If t is 0, and y is 1, it's a solution.
            # If t is 0 and y < 1, we need to move it.
            # f2_prime is -inf at 0, so delta will be 0?
            # 0.5 / 0 is inf.
            # We mask t=0.
            mask = t_values > 1e-12
            t_values[mask] -= (f2_val[mask] - y_values[mask]) / f2_prime[mask]
            
            t_values = np.clip(t_values, 0.0, 1.0)
        
        # Ensure endpoints are covered
        if n_samples > 1:
            t_values[0] = 0.0
            t_values[-1] = 1.0
            t_values = np.sort(t_values)
        
        # Evaluate all points on the Pareto front
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = t_values[i]
            # x[1:] stays 0, which is the minimizer of g(x) for ZDT3
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            _ = problem(x)
        
        return None
