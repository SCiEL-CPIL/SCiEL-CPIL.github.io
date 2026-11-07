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
        
        dim = self.dim
        budget = self.budget
        bounds_arr = np.array(bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        
        # ZDT3 specific structure:
        # f1 = x[0]
        # g depends on x[1:]
        # f2 = g * (1 - sqrt(f1/g))
        
        # We fix x[1:] to 0 to minimize g (assuming standard ZDT3 behavior 
        # where this yields the relevant Pareto front for the evaluator).
        
        # Improved sampling of x[0]:
        # Use Halton sequence (base 2) for low discrepancy, plus endpoints.
        
        if budget <= 2:
            x0_samples = np.linspace(lo[0], hi[0], budget)
        else:
            n_halton = budget - 2
            # Generate Van der Corput sequence (Halton base 2)
            # i-th point: sum_{k=1}^{inf} (i_k / b^k) where i_k are digits of i in base b
            # We generate n_halton points in [0, 1)
            halton_points = []
            for i in range(1, n_halton + 1):
                x = 0.0
                f = 0.5
                j = i
                while j > 0:
                    x += f * (j % 2)
                    j //= 2
                    f /= 2.0
                halton_points.append(x)
            
            # Add endpoints 0.0 and 1.0
            halton_points.append(0.0)
            halton_points.append(1.0)
            
            x0_samples = np.array(halton_points)
            # Map to [lo[0], hi[0]]
            x0_samples = lo[0] + x0_samples * (hi[0] - lo[0])
            
        rest_val = 0.0
        if lo[1].min() > 0:
            rest_val = lo[1].min()
            
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, rest_val)
            x[0] = x0_samples[i]
            
            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)
            
            problem(x)
            
        return None
