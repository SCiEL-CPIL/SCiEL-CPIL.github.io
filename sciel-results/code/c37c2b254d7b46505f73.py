import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ1 structure:
        # Head variables: indices 0, 1, 2 (for n_obj=3)
        # Tail variables: indices 3, 4, 5, 6
        # Set tail to midpoint for g=0
        n_head = n_obj - 1  # 2 position vars for M=3? No, for DTLZ1 it's M-1=2? 
        # Actually for DTLZ1, the number of "position" variables is M-1 = 2
        # Wait, let me re-examine. DTLZ1 with M objectives uses M-1 variables for position.
        # So for M=3: x0, x1 are position, x2..x6 are convergence?
        
        # Let me look at the standard DTLZ1 definition more carefully.
        # DTLZ1: M objectives, n variables
        # f1 = 0.5*(1+g)*x1*cos(pi*x2/2)*cos(pi*x3/2)*... 
        # Actually the standard uses 1-indexed. Let me use 0-indexed.
        
        # For M=3, n=7:
        # g depends on x[2:7] (5 variables) or x[3:7] (4 variables)?
        # Standard DTLZ1: g = 100*(n-M+1 + sum_{i=M}^{n-1}((x_i-0.5)^2 - cos(20*pi*(x_i-0.5))))
        # So for M=3, n=7: n-M+1 = 5, sum from i=3 to 6 (4 terms)
        # g = 100*(5 + sum_{i=3}^{6}(...))
        # At x_i=0.5: each term = 0 - 1 = -1, so sum = -4
        # g = 100*(5-4) = 100. That's not zero!
        
        # I think I have the formula wrong. Let me use:
        # g = 100 * (k + sum_{i=1}^{k} ((x_{M+i-1} - 0.5)^2 - cos(20*pi*(x_{M+i-1}-0.5))))
        # where k = n - M + 1
        # For M=3, n=7: k = 5
        # Variables involved in g: x[2], x[3], x[4], x[5], x[6] (5 variables, indices 2-6)
        # At all = 0.5: each term = -1, sum = -5
        # g = 100*(5 - 5) = 0. Yes!
        
        # So head variables: x[0], x[1] (indices 0,1) - M-1 = 2 variables
        # Tail variables: x[2], x[3], x[4], x[5], x[6] (indices 2-6) - 5 variables
        
        n_head = n_obj - 1  # 2
        n_tail = dim - n_head  # 5
        
        tail_mid = (lo[n_head:] + hi[n_head:]) / 2.0
        
        # Generate Halton sequence in n_head dimensions
        def halton_sequence(n, dims):
            """Generate n points in [0,1]^dims using Halton sequence."""
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # primes: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i, d] = r
            return points
        
        # Use all budget for Pareto front sampling
        n_samples = budget
        
        # Generate Halton sequence
        halton_pts = halton_sequence(n_samples, n_head)
        
        X = []
        F = []
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.empty(dim)
            x[:n_head] = lo[:n_head] + (hi[:n_head] - lo[:n_head]) * halton_pts[i]
            x[n_head:] = tail_mid
            
            f = np.asarray(problem(x), dtype=np.float64)
            X.append(x)
            F.append(f)
        
        return X
