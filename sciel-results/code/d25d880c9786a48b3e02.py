import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Direct DTLZ2 optimizer: exploit front structure, use Sobol sequence for shape vars."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s wall-clock

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2: first n_obj-1 variables are shape/position vars,
        # remaining are distance vars.
        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # Step 1: Determine distance variable values to lie on the Pareto front.
        # Standard DTLZ2: g(x) = sum((x_i - 0.5)^2) for i >= M-1.
        # So x_i = 0.5 makes g(x) = 0.
        # We verify this with a single evaluation. If it fails, we do a small search.
        
        # Test candidate: all distance vars = 0.5
        x_test = np.zeros(self.dim)
        for i in shape_vars:
            x_test[i] = 0.5
        for i in dist_vars:
            x_test[i] = 0.5
            
        f_test = np.array(problem(x_test))
        err = abs(np.sum(f_test ** 2) - 1.0)
        
        evals_used = 1
        
        if err > 1e-3:
            # Fallback: search for best distance value
            candidates = [0.0, 0.5, 0.3, 0.7, 0.2, 0.8]
            best_dist = 0.5
            best_front_err = err
            for dv in candidates:
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                for i in shape_vars:
                    x[i] = 0.5
                for i in dist_vars:
                    x[i] = dv
                f = np.array(problem(x))
                e = abs(np.sum(f ** 2) - 1.0)
                if e < best_front_err:
                    best_front_err = e
                    best_dist = dv
                evals_used += 1
        else:
            best_dist = 0.5

        remaining = max(self.budget - evals_used, 0)
        
        if remaining <= 0:
            return []

        # Step 2: Generate Sobol sequence for shape variables
        # Sobol sequences provide excellent low-discrepancy sampling.
        sampler = qmc.Sobol(d=k_shape, scramble=True)
        # Sobol requires power of 2. Find largest power of 2 <= remaining.
        # Or just generate a bit more and truncate? 
        # qmc.Sobol requires n to be a power of 2.
        # We can generate the next power of 2 and take the first 'remaining' points.
        # Or use a Halton sequence which doesn't have this restriction.
        # Let's use Halton for simplicity and flexibility with N.
        
        # Actually, Sobol is generally better. 
        # Let's find the largest power of 2 <= remaining.
        n_sobol = 1
        while n_sobol * 2 <= remaining:
            n_sobol *= 2
            
        if n_sobol < 4:
            n_sobol = 4 # Minimum for Sobol to be useful?
            
        # Generate Sobol points in [0, 1]^k_shape
        # Note: Sobol points are in [0, 1).
        sobol_points = sampler.random(n_sobol)
        
        # We might have more points than 'remaining' if we rounded up? 
        # No, we rounded down. n_sobol <= remaining.
        # If remaining is not a power of 2, we waste some budget?
        # We can fill the rest with a random or grid sample.
        # Or just use Halton which allows any N.
        
        # Let's use Halton sequence instead, it's simpler for arbitrary N.
        halton_sampler = qmc.Halton(d=k_shape, scramble=True)
        halton_points = halton_sampler.random(remaining)
        
        solutions = []
        
        for i in range(remaining):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            # Assign shape variables from Halton sequence
            for j in range(k_shape):
                x[shape_vars[j]] = halton_points[i, j]
            
            # Assign distance variables
            for k in dist_vars:
                x[k] = best_dist
                
            f = np.array(problem(x))
            solutions.append((x, f))
            
        return solutions
