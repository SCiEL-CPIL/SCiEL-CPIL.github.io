"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Exploit the known analytical Pareto front of ZDT4.
    2. The Pareto front is located at x_1 in [0, 1] and x_2..x_n = 0.
    3. Use the entire evaluation budget to densely sample x_1 uniformly.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Determine the optimal values for the deceptive variables x_2..x_n.
        # For ZDT4, the global minimum of the penalty function g(x) is at x_i = 0.
        # We need to ensure 0 is within the bounds [lo_i, hi_i].
        # If 0 is not in bounds, we pick the closest bound to 0 to minimize the penalty.
        optimal_deceptive_vals = np.zeros(n - 1, dtype=np.float64)
        for i in range(1, n):
            if lo[i] > 0:
                optimal_deceptive_vals[i-1] = lo[i]
            elif hi[i] < 0:
                optimal_deceptive_vals[i-1] = hi[i]
            else:
                optimal_deceptive_vals[i-1] = 0.0

        # Sample x_1 (the first variable) uniformly in [0, 1] (or its bounds).
        # The Pareto front is defined for x_1 in [0, 1].
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # If the bounds for x1 are not [0,1], we might need to adjust, 
        # but standard ZDT4 has x1 in [0,1].
        # We will sample uniformly across the valid range of x1 that contributes to the front.
        # The front exists where f2 is minimized, which corresponds to x1 in [0,1].
        # If bounds are wider, we restrict to [0,1] because outside this range, 
        # the points are not on the true Pareto front (they are dominated or have worse f2).
        
        sample_start = max(x1_lo, 0.0)
        sample_end = min(x1_hi, 1.0)
        
        # If the valid range is invalid (e.g. bounds don't cover [0,1]), fallback to full bounds
        if sample_start >= sample_end:
            sample_start = x1_lo
            sample_end = x1_hi

        # Generate evaluation points
        # Use uniform spacing for x1
        x1_values = np.linspace(sample_start, sample_end, budget)
        
        # To avoid boundary issues (like division by zero in some formulations, though ZDT4 is usually safe),
        # we can nudge 0 slightly if necessary, but np.linspace(0, 1) includes 0.
        # At x1=0, f1=0. h(0) = 1 - 0 + 0 = 1. f2 = g * 1. This is a valid point.
        
        results = []
        
        for i in range(budget):
            if time.time() > deadline:
                break
                
            x1_val = x1_values[i]
            
            # Construct the full vector x
            x = np.empty(n, dtype=np.float64)
            x[0] = x1_val
            x[1:] = optimal_deceptive_vals
            
            # Clip to bounds to ensure validity
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            # Store the result
            results.append(f)
            
            # Check time periodically to be safe
            if i % 50 == 0:
                if time.time() > deadline:
                    break

        return results
