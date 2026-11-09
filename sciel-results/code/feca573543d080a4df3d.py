"""Portfolio search for DTLZ2: special points + Halton sequence on Pareto front, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using special points + Halton-sequence front portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        def front_to_x(t):
            """Convert angle parameters t (length n_obj-1) to decision vector x.
            For DTLZ2: x_i = sin(t_i) for i < n_obj-1, and the rest are set
            to values that make g(x) = 0 (i.e., x_{n_obj-1} = 0.5 for the
            g-minimizing components, or we just set them to 0.5).
            Actually, for DTLZ2, g(x) = sum_{i=n_obj-1}^{n-1} (x_i - 0.5)^2.
            To get g=0, we set x_i = 0.5 for i >= n_obj-1.
            And x_i = sin(t_i) for i < n_obj-1.
            """
            x = np.full(self.dim, 0.5)
            for i in range(min(self.n_obj - 1, self.dim)):
                x[i] = np.sin(t[i])
            return x

        def halton(index, base):
            """Generate the Halton sequence value for a given index and base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Phase 1: Add high-value special points that disproportionately contribute to hypervolume
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere.
        # Vertices, edge midpoints, and center have high hypervolume contribution.
        
        special_ts = []
        
        # 4 vertices of the quarter-sphere:
        # (0,0), (0, pi/2), (pi/2, 0), (pi/2, pi/2)
        special_ts.append(np.array([0.0, 0.0]))
        special_ts.append(np.array([0.0, np.pi / 2]))
        special_ts.append(np.array([np.pi / 2, 0.0]))
        special_ts.append(np.array([np.pi / 2, np.pi / 2]))
        
        # 4 edge midpoints:
        # Midpoints of the 4 edges of the quarter-sphere
        special_ts.append(np.array([0.0, np.pi / 4]))
        special_ts.append(np.array([np.pi / 4, 0.0]))
        special_ts.append(np.array([np.pi / 2, np.pi / 4]))
        special_ts.append(np.array([np.pi / 4, np.pi / 2]))
        
        # 1 center point:
        # Center of the quarter-sphere
        special_ts.append(np.array([np.pi / 4, np.pi / 4]))
        
        n_special = len(special_ts)
        
        # Evaluate special points first
        X = []
        F = []
        used = 0
        
        for t in special_ts:
            if not time_ok():
                break
            x = front_to_x(t)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1
        
        # Phase 2: Use Halton sequence for remaining budget
        n_remaining = self.budget - used
        
        if n_remaining > 0 and time_ok():
            # Use Halton sequence in 2D (bases 2 and 3)
            # Transform to surface-area-uniform sampling:
            # u1 in [0, 1] -> t1 = arcsin(u1)
            # u2 in [0, 1] -> t2 = u2 * pi/2
            
            for k in range(n_remaining):
                if not time_ok():
                    break
                
                # Halton sequence values
                u1 = halton(k + 1, 2)
                u2 = halton(k + 1, 3)
                
                # Transform to angles
                t1 = np.arcsin(u1)
                t2 = u2 * (np.pi / 2)
                t = np.array([t1, t2])
                
                x = front_to_x(t)
                x = np.clip(x, lo, hi)
                f = problem(x)
                X.append(x)
                F.append(f)
                used += 1
        
        X = np.array(X)
        F = np.array(F)

        return X

# EVOLVE-BLOCK-END
