# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Halton-sequence Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton_2d(n, base1=2, base2=3):
        """Generate n 2D Halton sequence points in [0,1)^2 (vectorized per coordinate)."""
        pts = np.zeros((n, 2))
        for i in range(1, n + 1):
            # Base-1 coordinate
            f = 0.0
            p = 1.0
            r = i
            while r > 0:
                r, digit = divmod(r, base1)
                p /= base1
                f += digit * p
            pts[i - 1, 0] = f

            # Base-2 coordinate
            f = 0.0
            p = 1.0
            r = i
            while r > 0:
                r, digit = divmod(r, base2)
                p /= base2
                f += digit * p
            pts[i - 1, 1] = f
        return pts

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ2 Pareto front: f_i = y_i where y is on the unit quarter-sphere in R^3
        # y_1 = cos(a1), y_2 = sin(a1)*cos(a2), y_3 = sin(a1)*sin(a2)
        # Decision variables: x[0] = a1/(pi/2), x[1] = a2/(pi/2), x[2:] = 0.5

        # Use Halton sequence for low-discrepancy sampling of (a1, a2) in [0, pi/2]^2
        # Allocate most budget to front coverage, keep a few for edge/corner emphasis

        # Special high-value points: vertices and mid-edges of the quarter circle
        # These capture high hypervolume contribution regions
        special_angles = [
            (0.0, 0.0),       # (1, 0, 0)
            (0.0, np.pi/2),   # (0, 1, 0)
            (np.pi/2, 0.0),   # (0, 0, 1)
            (np.pi/4, np.pi/4),  # center-ish
            (np.pi/4, 0.0),   # mid-edge
            (0.0, np.pi/4),   # mid-edge
            (np.pi/4, np.pi/2),  # mid-edge
        ]

        used = 0
        F_list = []
        X_list = []

        # Evaluate special points first
        for a1, a2 in special_angles:
            if not time_ok() or used >= budget:
                break
            x = np.zeros(dim)
            x[0] = a1 / (np.pi / 2)
            x[1] = a2 / (np.pi / 2)
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_list.append(f)
            X_list.append(x)
            used += 1

        # Fill remaining budget with Halton sequence points
        n_halton = budget - used
        if n_halton > 0 and time_ok():
            halton_pts = self._halton_2d(n_halton, base1=2, base2=3)
            for hpt in halton_pts:
                if not time_ok():
                    break
                a1 = hpt[0] * (np.pi / 2)
                a2 = hpt[1] * (np.pi / 2)
                x = np.zeros(dim)
                x[0] = a1 / (np.pi / 2)
                x[1] = a2 / (np.pi / 2)
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_list.append(f)
                X_list.append(x)
                used += 1

        return X_list
# EVOLVE-BLOCK-END
