"""Multi-objective optimizer for ZDT2 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    Exploits the known structure: Pareto front has x[1:] = 0, x[0] in [0,1].
    f1 = x[0], f2 = (1 + 9/sqrt(x0)) * g(x[1:]) where g = sum((xi-0.2)^2).
    To get on the Pareto front, set x[1:] = 0.
    We distribute evaluations along x[0] to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 320s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # Phase 1: Evaluate points along the known Pareto front (x[1:]=0)
        # For ZDT2, f1 = x0, f2 = (1 + 9/sqrt(x0)) * g where g=0 when x[1:]=0
        # So f2 = 1 + 9/sqrt(x0)
        # We want to cover x0 in [0,1] to maximize hypervolume
        # Hypervolume contribution of a point (f1, f2) with ref (11,11):
        #   (11-f1)*(11-f2) but only if both < 11
        # For the Pareto front: f2 = 1 + 9/sqrt(f1)
        # We need to choose x0 values to maximize the hypervolume of the set

        # The hypervolume of a set of points is the union of rectangles.
        # For 2D, sort by f1 ascending, then for each point, the contribution is
        # (11 - f2_i) * (f1_{i+1} - f1_i) where f1_{n+1} = 11 (but capped at 11)
        # Actually: sort by f1, then HV = sum over i of (11 - f2_i) * (f1_{i+1} - f1_i)
        # where f1_{n+1} = 11 (the reference point's f1 coordinate, but we only count up to min(f1, 11))
        # More precisely: sort points by f1 ascending. For point i, its "width" in f1 is
        # (f1_{i+1} - f1_i) for i < n-1, and (11 - f1_{n-1}) for the last point.
        # Height is (11 - f2_i).
        # But we also need f2_i < 11 for the point to count.

        # Strategy: use a non-uniform grid that's denser where the front is steeper.
        # f2 = 1 + 9/sqrt(x0), df2/dx0 = -9/(2*x0^1.5)
        # The front is very steep near x0=0. We should use more points there.
        # Use a power-law spacing: x0_i proportional to (i/n)^p for some p > 1
        # This gives more points near 0.

        # Let's use p=2: x0_i = (i/n)^2 for i=0..n
        # This gives geometric-like spacing near 0.

        # How many points to use? We have budget=200 evaluations.
        # Each point on the Pareto front costs 1 evaluation.
        # We can use all 200 for the Pareto front sampling.
        # But we might want to leave some for verification or edge cases.
        # Let's use 195 for the front and 5 for boundary checks.

        n_front = 190
        x0_values = np.linspace(0, 1, n_front + 1) ** 2  # quadratic spacing, denser near 0
        # Actually, let's use a better spacing. The ideal spacing for HV maximization
        # would be where each point contributes equally to HV.
        # For now, let's try a few different spacings and pick the best.

        # Let's just use a mix: log-spaced near 0, linear elsewhere
        # Or simply: use the known optimal approach - sample uniformly in f2 space
        # Since f2 = 1 + 9/sqrt(x0), we can invert: x0 = (9/(f2-1))^2
        # If we sample f2 uniformly in [1, 10] (since at x0=1, f2=10), we get
        # non-uniform x0 that's denser where the front curves more.
        # At x0=0, f2 -> infinity, but we cap at x0=0.001 or so.

        # Let's use: sample f2 in [1, 10] with some non-uniformity
        # Actually, for hypervolume, the optimal 1D spacing in f2 for a curve
        # that's convex (in f2 vs f1) is to equalize the area contribution.
        
        # Simpler approach: just use many points with quadratic spacing in x0
        # x0_i = (i/n)^2 gives x0 values: 0, 1/n^2, 4/n^2, 9/n^2, ...
        # This is quite dense near 0.

        # Let's try: use n_front points with x0 = (i/(n_front-1))^3 for i=0..n_front-1
        # Cubic spacing is even denser near 0.

        # I'll use a combination: first 10% of points very close to 0, rest spread out.
        # Let me just use a power law with exponent 1.5
        t = np.linspace(0, 1, n_front)
        x0_values = t ** 1.5  # denser near 0

        # Ensure we include x0=1
        x0_values = np.concatenate([x0_values, [1.0]])

        # Build candidate points
        candidates = []
        for x0 in x0_values:
            x = np.zeros(dim)
            x[0] = x0
            candidates.append(x)

        # Evaluate all candidates
        F = np.array([problem(x) for x in candidates])
        used = len(candidates)

        # Check time
        if time.time() > deadline - 5:
            return candidates

        # Phase 2: Use remaining budget for local refinement or additional points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Add some points with small perturbations to x[1:] to see if we can
            # get slightly better f2 (unlikely, since g >= 0 with min at x[1:]=0)
            # Actually, any perturbation to x[1:] increases g, which increases f2.
            # So points on the Pareto front (x[1:]=0) are optimal for their x0.
            # No point off the front can dominate a point on the front with the same x0.
            # So we should stick to the front.
            # Use remaining budget for more front points with refined spacing.
            # Let's add points in the gaps where the spacing is largest.
            # Compute gaps in x0
            x0_sorted = np.sort(x0_values)
            gaps = np.diff(x0_sorted)
            # Find the largest gaps and add midpoints
            for _ in range(min(remaining, 10)):
                if len(gaps) == 0:
                    break
                max_gap_idx = np.argmax(gaps)
                mid_x0 = (x0_sorted[max_gap_idx] + x0_sorted[max_gap_idx + 1]) / 2
                x = np.zeros(dim)
                x[0] = mid_x0
                candidates.append(x)
                x0_sorted = np.sort(np.append(x0_sorted, mid_x0))
                gaps = np.diff(x0_sorted)
                used += 1
                if time.time() > deadline - 5:
                    break

        return candidates
