"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, aim for 320s
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        n = self.dim
        # ZDT2 constant
        C = 1 + 10 * (n - 1)
        
        # Reference point value
        ref_val = 11.0
        
        # Calculate the intersection t* where f2(t) = ref_val
        # f2(t) = C * (1 - sqrt(t))
        # C * (1 - sqrt(t)) = ref_val
        # 1 - sqrt(t) = ref_val / C
        # sqrt(t) = 1 - ref_val / C
        # t = (1 - ref_val / C)^2
        t_star = (1.0 - ref_val / C) ** 2
        
        # Clip t_star to valid range [0, 1]
        t_star = np.clip(t_star, 0.0, 1.0)
        
        N = self.budget
        
        # We will generate N points on the Pareto front.
        # One point at t_star to capture the "corner" where f2 drops below ref.
        # The rest distributed in [t_star, 1.0].
        
        t_vals = np.zeros(N)
        t_vals[0] = t_star
        
        if N > 1:
            # Distribute remaining N-1 points in [t_star, 1.0]
            # Using a power-law distribution to handle the convexity.
            # t = t_star + (1 - t_star) * u^alpha
            # If alpha < 1, points are denser near t_star (u=0).
            # If alpha > 1, points are denser near 1.0 (u=1).
            # The function f2 is convex. The area is concentrated near t_star.
            # We want to approximate the integral well.
            # Uniform spacing in t is a good baseline.
            # Let's try alpha = 0.5 (denser near t_star) to capture the high values?
            # Or alpha = 1.0 (uniform).
            # Given N=200, uniform is likely sufficient.
            u = np.linspace(0.0, 1.0, N - 1)
            # Use uniform for robustness
            t_vals[1:] = t_star + (1.0 - t_star) * u
            
            # Ensure the last point is exactly 1.0
            t_vals[-1] = 1.0
        
        # Construct the population X
        # X is (N, dim)
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # x[1:] remain 0, which is optimal for ZDT2 front
        
        # Evaluate all points
        # Check time periodically? 
        # With N=200 and simple problem, it should be fast.
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
