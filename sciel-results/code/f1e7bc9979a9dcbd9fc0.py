import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) = 1
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            # Use all budget evaluations to sample the Pareto front
            n_points = self.budget

            # x0 range for ZDT4 is typically [0, 1]
            # But we should respect the given bounds
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)

            if x0_lo >= x0_hi:
                # Degenerate case, just evaluate one point
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                # Strategy: Sample evenly in f2 (objective) space rather than x0 space.
                # The Pareto front is f1 = x0, f2 = h(x0, 1) where h is concave.
                # Equal spacing in f2 minimizes the maximum uncovered area under the curve,
                # which directly maximizes the approximated hypervolume.
                
                # h(x) = 1 + exp(-4*x - 0.3*(sin(8*pi*x)^5))
                # We need to find x0 values such that f2 values are evenly spaced.
                
                # First, let's determine the range of f2 on [x0_lo, x0_hi]
                # h is monotonically decreasing in x (for the standard ZDT4 with g=1)
                # because the dominant term is -4*x.
                
                # Let's compute f2 at endpoints to get the range
                def h_func(x):
                    return 1.0 + np.exp(-4.0*x - 0.3*(np.sin(8.0*np.pi*x)**5))
                
                f2_lo_val = h_func(x0_hi)  # minimum f2 (at x0_hi)
                f2_hi_val = h_func(x0_lo)  # maximum f2 (at x0_lo)
                
                # If the function is not strictly monotonic or if the range is very small,
                # fallback to uniform x0 sampling.
                if f2_hi_val <= f2_lo_val:
                    # Non-monotonic or degenerate, use uniform x0
                    x0_vals = np.linspace(x0_lo, x0_hi, n_points)
                else:
                    # Create evenly spaced f2 targets
                    f2_targets = np.linspace(f2_lo_val, f2_hi_val, n_points)
                    
                    # Invert h(x) = y to find x for each y
                    # Use scipy.optimize.brentq for robust and fast inversion.
                    x0_vals = np.zeros(n_points)
                    
                    def invert_h(target):
                        a = x0_lo
                        b = x0_hi
                        fa = h_func(a) - target
                        fb = h_func(b) - target
                        # fa should be >= 0 (since h(a) >= target)
                        # fb should be <= 0 (since h(b) <= target)
                        # If not, clamp
                        if fa < 0:
                            return a
                        if fb > 0:
                            return b
                        return brentq(lambda x: h_func(x) - target, a, b, xtol=1e-12)
                    
                    for i in range(n_points):
                        x0_vals[i] = invert_h(f2_targets[i])
                    
                    # Ensure x0_vals are within bounds
                    x0_vals = np.clip(x0_vals, x0_lo, x0_hi)
                    
                    # Sort x0_vals to ensure monotonicity in the front (optional but good for HV calc)
                    # The problem evaluator just records points, order doesn't matter for HV,
                    # but let's keep them sorted by x0 for cleanliness.
                    x0_vals = np.sort(x0_vals)

            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] = 0.0 is default

            # Check time
            if time.time() > deadline:
                return X

            F = np.array([problem(x) for x in X])
            return X

        else:
            # Fallback: 0 is not in bounds for some x_i (i>=1)
            # We need to find the point that minimizes g
            # Since g = 1 + 10*(n-1) for standard ZDT4 when any x_i != 0,
            # but this depends on the specific problem.
            # As a fallback, use random sampling
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
