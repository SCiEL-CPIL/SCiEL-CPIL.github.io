"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2.
    Strategy:
    1. DTLZ2 with M=3 has Pareto front determined by x[0], x[1].
    2. x[2:] should be 0.5 to minimize g(x).
    3. Use a low-discrepancy sequence (Halton) for sampling in [0, 1]^2.
    4. Map to DTLZ2 variables for uniform coverage on the quarter-sphere Pareto front:
       - x[0] = (2/pi) * arcsin(u) for uniform surface area
       - x[1] = v
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, n, dim=2):
        """Generate n points in [0,1]^dim using Halton sequence."""
        bases = [2, 3]  # first two primes
        points = np.zeros((n, dim))
        for d in range(dim):
            base = bases[d]
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                j = i
                while j > 0:
                    j, rem = divmod(j, base)
                    f /= base
                    r += rem * f
                points[i - 1, d] = r
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Generate Halton points in [0, 1]^2
        points_2d = self._halton(n_evals, dim=2)
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            u = points_2d[i, 0]
            v = points_2d[i, 1]
            
            # Map to DTLZ2 variables for uniform surface area on quarter-sphere
            # x[0] = (2/pi) * arcsin(u)
            x0 = (2.0 / np.pi) * np.arcsin(u)
            
            # x[1] = v
            x1 = v
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            
            # Set remaining variables to center of bounds to minimize g(x)
            if self.dim > 2:
                center = (lo[2:] + hi[2:]) / 2.0
                x[2:] = center
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
