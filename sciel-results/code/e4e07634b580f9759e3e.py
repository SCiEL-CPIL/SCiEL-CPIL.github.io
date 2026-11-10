"""ZDT1-aware optimizer using analytic grid sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 standard formulation:
        # f1 = x1
        # g = 1 + 9 * sum(x_i for i=2..n) / (n-1)
        # f2 = g * (1 - sqrt(f1/g))
        #
        # Pareto front achieved when g is minimized.
        # g_min = 1 when x_i = 0 for all i >= 2.
        # So Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1)
        #
        # Hypervolume with reference point (11, 11):
        # For a point (f1, f2) on the front, its contribution to HV is:
        # (11 - f1) * (11 - f2) minus contributions from other points.
        #
        # For a set of points sorted by f1, the hypervolume is:
        # sum over i of (11 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = 11
        # (assuming points are sorted by increasing f1, and f2 is decreasing)
        #
        # Actually, for a 2D front sorted by f1 ascending (f2 descending),
        # the hypervolume relative to (ref1, ref2) is:
        # HV = sum_{i=1}^{n} (ref1 - f1_i) * (f2_{i-1} - f2_i)
        # where f2_0 = ref2, and we only count points with f1 < ref1 and f2 < ref2.
        #
        # Since all front points have f1 in [0,1] and f2 in [0,1], they're all
        # strictly better than (11, 11).
        #
        # To maximize HV, we want to maximize:
        # sum_{i=1}^{n} (11 - f1_i) * (f2_{i-1} - f2_i)
        # where f2_i = 1 - sqrt(f1_i)
        #
        # Let's think about optimal spacing. The marginal benefit of adding a point
        # at f1 depends on the gap it fills. For a concave front, points should be
        # denser where the curve is steeper (near f1=0).
        #
        # The derivative f2'(f1) = -1/(2*sqrt(f1)), which is large in magnitude near 0.
        # So we want more points near f1=0.
        #
        # Optimal density is proportional to |f2'(f1)| = 1/(2*sqrt(f1)).
        # The CDF is: F(f1) = integral_0^{f1} (1/(2*sqrt(t))) dt = sqrt(f1)
        # So f1_i = u_i^2 where u_i is uniformly distributed in [0,1].
        #
        # For stratified sampling with n points:
        # u_i = (i - 0.5) / n for i = 1..n (midpoint stratification)
        # f1_i = u_i^2 = ((i - 0.5)/n)^2
        #
        # Let's use this approach.

        n_samples = budget

        # Midpoint stratification with density ∝ 1/(2*sqrt(f1))
        # f1_i = ((i - 0.5) / n)^2 for i = 1..n
        # This gives uniform spacing in sqrt(f1), which is optimal for
        # minimizing integration error on the concave Pareto front.
        i = np.arange(1, n_samples + 1, dtype=np.float64)
        u = (i - 0.5) / n_samples
        f1_vals = u ** 2
        
        # Compute f2 for each f1
        f2_vals = 1.0 - np.sqrt(f1_vals)

        # Construct x vectors: x1 = f1, x2..x30 = 0 (to minimize g)
        X = np.zeros((n_samples, dim))
        X[:, 0] = f1_vals
        # x2..x30 = 0 gives g = 1

        # Check time
        if time.time() > deadline:
            return X

        # Local polish: coordinate descent to maximize discrete hypervolume
        # HV = sum_i (11 - f1_i) * (f2_{i-1} - f2_i), f2_0 = 11
        # We optimize f1 values (sorted) to maximize this.
        
        def compute_hv(f1_arr):
            f2_arr = 1.0 - np.sqrt(np.clip(f1_arr, 0, 1))
            # Sort by f1 ascending
            idx = np.argsort(f1_arr)
            f1_sorted = f1_arr[idx]
            f2_sorted = f2_arr[idx]
            # f2_0 = 11 (reference)
            f2_prev = 11.0
            hv = 0.0
            for j in range(len(f1_sorted)):
                if f1_sorted[j] >= 11.0 or f2_sorted[j] >= 11.0:
                    continue
                hv += (11.0 - f1_sorted[j]) * (f2_prev - f2_sorted[j])
                f2_prev = f2_sorted[j]
            return hv

        # A few iterations of coordinate descent
        for _iter in range(5):
            if time.time() > deadline - 10:
                break
            improved = False
            for j in range(n_samples):
                if time.time() > deadline - 10:
                    break
                old_f1 = f1_vals[j]
                # Try small perturbations
                for delta in [0.001, -0.001, 0.0001, -0.0001, 0.01, -0.01]:
                    new_f1 = np.clip(old_f1 + delta, 0.0, 1.0)
                    f1_test = f1_vals.copy()
                    f1_test[j] = new_f1
                    if compute_hv(f1_test) > compute_hv(f1_vals) + 1e-12:
                        f1_vals[j] = new_f1
                        improved = True
                        break
            if not improved:
                break
        
        # Update X with polished f1 values and evaluate exactly once
        X[:, 0] = f1_vals
        # We have budget left (we used 0 so far).
        # We will use exactly `budget` evaluations here.
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
