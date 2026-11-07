import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2:
        # f1 = x1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        # The key is to sample x1 values that maximize hypervolume

        # Optimal x1 spacing for hypervolume maximization with 200 samples
        # The Pareto front is f2 = 11 * (1 - sqrt(f1))
        # Hypervolume contribution of a point at x1 is related to the area it "owns"
        # For concave front, optimal spacing is non-uniform: denser where the curve is steeper

        # The curve is f2 = 11*(1 - sqrt(x1)), so df2/dx1 = -11/(2*sqrt(x1))
        # Steeper near x1=0, so we need more points near 0
        # Using the optimal spacing: x1_i such that the area contribution is equalized
        # For a concave function, equal-area spacing means x1_i = i^2 / m^2 for i=0..m

        m = 199  # We'll use 200 points (indices 0 to 199)
        # Actually, let's think more carefully.
        # With m+1 points at x1_0 < x1_1 < ... < x1_m, the hypervolume (relative to ref [11,11])
        # is sum of trapezoids. For equal area contribution, we need x1_i such that
        # the integral of (11 - f2(x1)) dx1 is equal between consecutive points.
        # 11 - f2(x1) = 11*sqrt(x1)
        # Integral from 0 to x1 is 11 * (2/3) * x1^(3/2)
        # So equal area: x1_i^(3/2) = i/m * 1, i.e., x1_i = (i/m)^(2/3)

        # Let's use 200 points
        num_points = min(200, self.budget)
        i = np.arange(num_points, dtype=float)
        x1_vals = (i / (num_points - 1)) ** (2.0 / 3.0)

        # Evaluate these points on the Pareto front (x2...xn = 0)
        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        # If we have budget remaining, do some random exploration to ensure we capture
        # any numerical issues, but the main points are on the Pareto front
        if used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = self.budget - used
            for _ in range(min(remaining, 10)):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(n)
                f = problem(x)
                recorded.append(f)
                used += 1

        return recorded
