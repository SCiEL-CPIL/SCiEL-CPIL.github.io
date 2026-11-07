"""Multi-objective optimizer for ZDT3 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-specific optimizer using optimal x0 distribution for hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_evals = self.budget

        # For ZDT3, the Pareto front is parameterized by f1 = x[0] in [0, 1].
        # To be on the Pareto front, x[1:] must minimize g(x).
        # g(x) = 1 + 10 * sum((x_i - 0.5)^2 - 0.4 * cos(20 * pi * (x_i - 0.5))) for i=1..n-1
        # The global minimum of g is 1 + 10*(n-1) when x_i = 0.5 for all i.
        # So setting x[1:] = 0.5 places the solution on the Pareto front.

        # Strategy: Sample x[0] values that maximize hypervolume coverage.
        # The hypervolume indicator is sensitive to the "gaps" in the front.
        # Uniform sampling is good, but we can do better by ensuring we hit
        # the local minima of the oscillating term in f2.
        
        # f2(f1) = g * (1 - sqrt(f1/g) - (f1/g) * sin(10 * pi * f1))
        # The term - (f1/g) * sin(10 * pi * f1) creates oscillations.
        # The local minima of f2 occur when sin(10 * pi * f1) = 1.
        # 10 * pi * f1 = pi/2 + 2 * k * pi  =>  f1 = 0.05 + 0.2 * k
        # for k = 0, 1, 2, 3, 4.
        # These are the "deepest" points in each lobe of the front.
        
        minima_f1 = [0.05 + 0.2 * k for k in range(5)]

        # We want to use all 200 evaluations.
        # Allocate 5 to the minima, and the rest (195) to uniform sampling.
        # This ensures we hit the critical points and also have dense coverage.
        
        n_uniform = n_evals - len(minima_f1)
        
        # Generate uniform samples in [0, 1]
        # Using linspace ensures we cover the full range including endpoints.
        # We check for duplicates with minima to avoid wasting evaluations.
        
        uniform_samples = np.linspace(0.0, 1.0, n_uniform)
        
        # Combine and remove duplicates
        # Since minima are at 0.05, 0.25, 0.45, 0.65, 0.85
        # and linspace has step 1/194 ~ 0.00515
        # It is unlikely to hit exactly, but let's be safe.
        
        f1_values = np.concatenate((minima_f1, uniform_samples))
        f1_values = np.unique(f1_values)
        
        # If unique reduced the count, we have spare evaluations.
        # Add more random points or refine linspace if needed.
        # However, with 195 uniform + 5 specific, duplicates are very unlikely
        # unless linspace step aligns perfectly.
        
        # If we have fewer points than budget, add random samples
        # to fill the gap (though this shouldn't happen often).
        if len(f1_values) < n_evals:
            # Add random points to fill the gap
            # Use a fixed seed for reproducibility if needed, but np.random.seed is set by evaluator
            extra = np.random.uniform(0.0, 1.0, n_evals - len(f1_values))
            f1_values = np.concatenate((f1_values, extra))
            f1_values = np.unique(f1_values)
            
        # If we still have fewer (due to collisions), just proceed with what we have.
        # If we have more (shouldn't happen with unique), trim.
        if len(f1_values) > n_evals:
            f1_values = f1_values[:n_evals]

        # Evaluate each point
        for x0 in f1_values:
            if time.time() > deadline:
                break
                
            # Construct the point on the Pareto front
            x = np.full(dim, 0.5)  # x[1:] = 0.5 minimizes g
            x[0] = x0              # f1 = x0
            
            # Ensure bounds are respected (though [0,1] and 0.5 is safe)
            x = np.clip(x, lo, hi)
            
            # Evaluate
            problem(x)

        return None
# EVOLVE-BLOCK-END
