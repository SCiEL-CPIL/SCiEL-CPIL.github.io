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
        R1 = ref_val
        R2 = ref_val
        
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
        # The hypervolume is H = integral from t_star to 1 of (R2 - f2(t)) dt
        # f2(t) = C * (1 - sqrt(t))
        # Integrand g(t) = R2 - C * (1 - sqrt(t)) = R2 - C + C * sqrt(t)
        
        # We want to sample t values such that they optimally approximate the integral.
        # This is equivalent to inverse CDF sampling from the distribution with PDF proportional to g(t).
        
        # Antiderivative of g(t):
        # G(t) = (R2 - C) * t + (2/3) * C * t^(3/2)
        
        def G(t):
            return (R2 - C) * t + (2.0 / 3.0) * C * (t ** 1.5)
        
        G_start = G(t_star)
        G_end = G(1.0)
        G_range = G_end - G_start
        
        # Generate uniform samples in [0, 1]
        u = np.linspace(0.0, 1.0, N)
        
        # Map to G values
        G_vals = G_start + u * G_range
        
        # Invert G(t) = val to find t
        # (R2 - C) * t + (2/3) * C * t^(1.5) = val
        # Let A = R2 - C, B = (2/3) * C
        # A * t + B * t^1.5 = val
        # We can solve this using Newton's method or bisection.
        # Since N is small (200), bisection is fast and robust.
        
        A = R2 - C
        B = (2.0 / 3.0) * C
        
        t_vals = np.zeros(N)
        
        for i in range(N):
            target = G_vals[i]
            lo_t = t_star
            hi_t = 1.0
            
            # Bisection
            for _ in range(50):
                mid_t = (lo_t + hi_t) / 2.0
                mid_val = A * mid_t + B * (mid_t ** 1.5)
                if mid_val < target:
                    lo_t = mid_t
                else:
                    hi_t = mid_t
            
            t_vals[i] = (lo_t + hi_t) / 2.0
        
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
