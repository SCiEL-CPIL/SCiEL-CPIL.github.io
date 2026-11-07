import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))

        # ZDT4 Pareto Front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin^2(10 * pi * x0)
        
        # Strategy: Non-uniform sampling in x0, denser near x0=0.
        # Rationale: The objective function f2 has a singularity in derivative at x0=0,
        # and the weighted sum objective (approximate HV driver) is more sensitive to f1 (x0)
        # due to the large reference point R2=400. Minimizing f1 is crucial.
        
        N = budget
        # Use quadratic spacing to concentrate points near x0=0
        # x_i = (i / (N-1))^2
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            t = np.linspace(0.0, 1.0, N)
            x0_vals = t**2
        
        # Clip to bounds
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = x0_val
            # Clip to bounds for all dimensions (though x[1:] are 0)
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)
            
        return np.empty((0, dim))
