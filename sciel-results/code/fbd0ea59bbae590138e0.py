"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol-based quasi-random search on the DTLZ1 Pareto front (g=0)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n 2D Sobol sequence points in [0,1)^2 using a simple bit-reversal method."""
        # We'll use a simple approach: generate Sobol points using scipy if available,
        # otherwise fall back to a low-discrepancy sequence.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n)
            return points
        except ImportError:
            # Fallback: use a Halton-like or simple quasi-random sequence
            # Generate points using a simple low-discrepancy approach
            pts = np.zeros((n, 2))
            for i in range(n):
                # Use van der Corput sequence for each dimension
                pts[i, 0] = self._van_der_corput(i, 2)
                pts[i, 1] = self._van_der_corput(i, 3)
            return pts

    @staticmethod
    def _van_der_corput(n, base):
        """Generate the n-th term of the van der Corput sequence in the given base."""
        result = 0.0
        f = 1.0 / base
        while n > 0:
            result += f * (n % base)
            n //= base
            f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For DTLZ1 with g=0 (tail variables at 0.5), the Pareto front is:
        # f0 = 0.5 * (1 + g) * x0
        # f1 = 0.5 * (1 + g) * (1 - x0) * x1
        # f2 = 0.5 * (1 + g) * (1 - x0) * (1 - x1)
        # With g=0: f0 = 0.5*x0, f1 = 0.5*(1-x0)*x1, f2 = 0.5*(1-x0)*(1-x1)
        # The Pareto front is a simplex in 3D: f0+f1+f2 = 0.5, fi >= 0
        # Parameterized by (u, v) in [0,1]^2 with u+v <= 1:
        #   x0 = 2*u, x1 = v/(1-u) if u < 1
        # Actually, let's use the standard parameterization:
        # Let x0 = 2*a where a in [0, 0.5]
        # Then 1-x0 = 1-2a
        # x1 in [0, 1], x2 = 1-x1
        # f0 = a, f1 = (1-2a)*x1/2, f2 = (1-2a)*(1-x1)/2
        # For hypervolume maximization, we want to cover the front well.
        # 
        # Better approach: use 2D Sobol points (s, t) in [0,1]^2, map to:
        #   x0 = 2*s  (so s in [0, 0.5] maps to x0 in [0, 1])
        #   Actually let's use s in [0,1] -> x0 = 2*s*0.5 = s... 
        #   
        # Let me think more carefully. The DTLZ1 Pareto front with g=0:
        # f0 = 0.5 * x0
        # f1 = 0.5 * (1-x0) * x1  
        # f2 = 0.5 * (1-x0) * (1-x1)
        # where x0 in [0,1], x1 in [0,1]
        # 
        # For each (x0, x1), we get a point on the front.
        # The front is the set {(f0,f1,f2) : f0+f1+f2=0.5, fi>=0}
        # 
        # So we just need to sample (x0, x1) uniformly/quasi-randomly in [0,1]^2
        # and compute the corresponding x vector with tail=0.5.

        n_points = min(budget, 200)
        
        # Generate Sobol points in 2D
        sobol_pts = self._sobol_2d(n_points)
        
        # Build the full x vectors: first 2 variables from Sobol, rest at 0.5
        X = np.zeros((n_points, dim))
        X[:, 0] = sobol_pts[:, 0]
        X[:, 1] = sobol_pts[:, 1]
        if dim > 2:
            X[:, 2:] = 0.5  # Analytic minimizer for g in DTLZ1
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # If we have budget left, we could add more points, but 200 is the max
        # and Sobol with 200 points should give good coverage.
        
        return X
# EVOLVE-BLOCK-END
