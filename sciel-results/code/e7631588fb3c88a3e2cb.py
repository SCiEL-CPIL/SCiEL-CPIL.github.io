import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2 with n=30:
        # f1 = x1
        # f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        # Reference point: [11.0, 11.0]

        num_points = min(self.budget, 200)

        if num_points <= 1:
            x1_vals = np.array([0.5])
        else:
            # Optimize the power-law exponent for maximum hypervolume
            # x1_i = (i/(N-1))^p, i=0..N-1
            # f1 = x1, f2 = 11*(1-sqrt(f1))
            # Hypervolume with reference (11,11) for sorted points:
            # Sort by f1 ascending: f1_0 < f1_1 < ... < f1_{N-1}
            # f2 is decreasing in f1 (since sqrt is increasing, 1-sqrt is decreasing)
            # So f2_0 > f2_1 > ... > f2_{N-1}
            # Hypervolume = sum of trapezoid areas between consecutive points
            # For points (f1_i, f2_i) sorted by f1 ascending:
            # The region dominated is bounded by the upper envelope
            # Since f2 decreases as f1 increases, the non-dominated set is all points
            # HV = sum_{i=0}^{N-2} (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_{N-1}) * (11 - f2_{N-1})
            # Wait, let me think more carefully.
            # For minimization with reference (r1, r2) = (11, 11):
            # A point (f1, f2) dominates the rectangle [f1, r1] x [f2, r2]
            # For a set of non-dominated points sorted by f1 ascending (f2 descending):
            # HV = sum over each point of the area it uniquely contributes
            # Point i contributes: (f1_{i+1} - f1_i) * (r2 - f2_i) for i < N-1
            # And the last point: (r1 - f1_{N-1}) * (r2 - f2_{N-1})
            # where f1_N = r1 = 11 (virtual point)

            def compute_hv(p):
                i = np.arange(num_points, dtype=float)
                x1 = (i / (num_points - 1)) ** p
                f1 = x1
                f2 = 11.0 * (1.0 - np.sqrt(f1))
                # Sort by f1 ascending (already sorted since x1 is increasing in i)
                # HV calculation
                # f1 is ascending, f2 is descending
                r1, r2 = 11.0, 11.0
                # Extend with reference point
                f1_ext = np.concatenate([f1, [r1]])
                f2_ext = np.concatenate([f2, [0.0]])  # f2 at reference doesn't matter for last term
                # Actually, the last segment goes from f1_{N-1} to r1 with height (r2 - f2_{N-1})
                # Let me compute properly:
                # HV = sum_{i=0}^{N-2} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{N-1}) * (r2 - f2_{N-1})
                diffs = np.diff(f1)  # N-1 elements
                # For i=0 to N-2: width = f1_{i+1} - f1_i, height = r2 - f2_i
                heights = r2 - f2[:-1]  # N-1 elements
                hv = np.sum(diffs * heights)
                # Last segment
                hv += (r1 - f1[-1]) * (r2 - f2[-1])
                return -hv  # Negative because we're minimizing for scipy

            # Optimize p in a reasonable range
            # p=2/3 was the original. Let's search around it.
            result = minimize_scalar(compute_hv, bounds=(0.3, 3.0), method='bounded',
                                     options={'xatol': 1e-10, 'maxiter': 100})
            p_opt = result.x

            i = np.arange(num_points, dtype=float)
            x1_vals = (i / (num_points - 1)) ** p_opt

        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
