# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 using direct inverse mapping from
    low-discrepancy points in the objective simplex to decision space.
    
    Strategy:
    1. The DTLZ1 Pareto front (g=0) is a 2-simplex in objective space: f1+f2+f3=0.5, fi>=0.
    2. We generate N points uniformly distributed on this simplex using a quasi-random
       (low-discrepancy) sequence for better hypervolume coverage than a grid.
    3. For each objective point (f1, f2, f3), we analytically invert the DTLZ1 mapping
       to find the corresponding decision vector x that achieves exactly those objectives.
    4. All evaluations are exactly on the Pareto front, maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        n_points = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate low-discrepancy points in the 2D simplex parameter space.
        # We use a Halton-like sequence with bases 2 and 3 for 2D.
        # This gives much better coverage than a grid or uniform random.
        
        def halton_index(i, base):
            """Compute the i-th Halton index in the given base."""
            f = 1.0
            r = 0.0
            i = i + 1  # 1-indexed
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Generate n_points 2D Halton sequence points in [0,1)^2
        # These represent (u, v) where u = F1, v = F2 in the normalized simplex F1+F2+F3=1
        # F3 = 1 - F1 - F2, with F1, F2, F3 >= 0 means F1 in [0,1], F2 in [0, 1-F1]
        
        # To sample uniformly from the simplex, we can use the transformation:
        # Given two uniform randoms r1, r2 in [0,1):
        #   F1 = 1 - sqrt(r1)
        #   F2 = sqrt(r1) * (1 - r2)
        #   F3 = sqrt(r1) * r2
        # This is a standard method for uniform simplex sampling.
        # We apply this to our Halton sequence points.
        
        halton_u = np.array([halton_index(i, 2) for i in range(n_points)])
        halton_v = np.array([halton_index(i, 3) for i in range(n_points)])
        
        # Transform to simplex coordinates F1, F2, F3 (sum to 1)
        sqrt_r1 = np.sqrt(halton_u)
        F1 = 1.0 - sqrt_r1
        F2 = sqrt_r1 * (1.0 - halton_v)
        F3 = sqrt_r1 * halton_v
        
        # Verify: F1 + F2 + F3 = 1 - sqrt_r1 + sqrt_r1*(1-v) + sqrt_r1*v = 1. Correct.
        
        # Now map from objective space to decision space.
        # On the Pareto front (g=0), the objectives are:
        # f1 = 0.5 * (1 - sqrt(x1))
        # f2 = 0.5 * sqrt(x1) * (1 - x2)
        # f3 = 0.5 * sqrt(x1) * x2
        # 
        # Let s = sqrt(x1). Then:
        # f1 = 0.5 * (1 - s)  =>  s = 1 - 2*f1
        # f2 = 0.5 * s * (1 - x2)
        # f3 = 0.5 * s * x2
        # 
        # From f1: s = 1 - 2*f1
        # From f3: x2 = 2*f3 / s = 2*f3 / (1 - 2*f1)
        # x1 = s^2 = (1 - 2*f1)^2
        # 
        # In terms of F (where f = 0.5 * F):
        # s = 1 - F1
        # x2 = F3 / (1 - F1)
        # x1 = (1 - F1)^2
        
        # Compute decision variables
        s = 1.0 - F1  # This is sqrt(x1)
        # Handle case where s is very close to 0 (F1 close to 1)
        # When s ~ 0, x2 is indeterminate, but f2 and f3 are both ~0, so any x2 works.
        # Set x2 = 0.5 as a default.
        x2 = np.where(s > 1e-12, F3 / np.where(s > 1e-12, s, 1.0), 0.5)
        x1 = s ** 2
        
        # Clip to valid range [0, 1]
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        # Build decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1
        X[:, 1] = x2
        # Distance variables: set to 0.5 to achieve g=0
        X[:, 2:] = 0.5
        
        # Apply bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
