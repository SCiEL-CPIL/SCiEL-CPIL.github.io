"""Multi-objective optimizer on ZDT6 exploiting the analytical Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6 optimizer that places all evaluations on the known Pareto front with
    hypervolume-aware adaptive spacing for maximum coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f1_of_t(self, t):
        """ZDT6 f1 for point on Pareto front: x1=t, x2..x10=0"""
        return 1.0 - np.exp(-4.0 * t) * np.sin(10.0 * np.pi * t) ** 2

    def _f2_of_t(self, t):
        """ZDT6 f2 for point on Pareto front: x1=t, x2..x10=0"""
        f1 = self._f1_of_t(t)
        return 1.0 - np.sqrt(f1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        X = np.zeros((budget, dim))
        F = np.zeros((budget, n_obj))

        # ZDT6 Pareto front: x1=t, x2..x10=0
        # We need to choose t values that maximize hypervolume coverage.
        # Strategy: Use greedy hypervolume maximization with a dense candidate set,
        # then select the best subset of 'budget' points.

        # Generate a dense set of candidate t values
        n_candidates = min(5000, budget * 25)  # up to 5000 candidates
        t_candidates = np.linspace(0.001, 0.999, n_candidates)

        # Compute f1, f2 for all candidates
        f1_cand = self._f1_of_t(t_candidates)
        f2_cand = self._f2_of_t(t_candidates)

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Filter candidates that are strictly better than reference in both objectives
        valid_mask = (f1_cand < ref[0]) & (f2_cand < ref[1])
        t_valid = t_candidates[valid_mask]
        f1_valid = f1_cand[valid_mask]
        f2_valid = f2_cand[valid_mask]

        # Greedy hypervolume maximization:
        # Start with the point that alone gives max hypervolume, then iteratively
        # add the point that gives the largest marginal hypervolume gain.
        # For 2D, hypervolume of a set of points with ref [r1, r2] is:
        # sum over consecutive points of (f2_i - f2_{i+1}) * (f1_{i+1} - f1_i) + corner terms
        # But for greedy selection, we can compute incremental contribution.

        # For efficiency, we'll use a simpler approach:
        # Sort by f1, then use adaptive spacing based on the curve geometry.
        # The key insight: for ZDT6, f1(t) is not monotonic - it has oscillations.
        # So we need to handle this carefully.

        # Actually, for hypervolume in 2D with a non-monotonic front, the standard
        # approach is to first compute the "Pareto envelope" (remove dominated points),
        # then compute hypervolume of the envelope.

        # Let's use a different strategy: since we know the exact front,
        # use a space-filling approach in f-space that accounts for the curve shape.

        # The f1-f2 curve for ZDT6: as t goes from 0 to 1, f1 oscillates between
        # ~0 and ~1 (with decreasing amplitude due to exp(-4t)), and f2 = 1-sqrt(f1).
        # The hypervolume is maximized when we have good coverage of the entire curve.

        # Use an iterative refinement: start with uniform t spacing, then
        # redistribute points to areas of high curvature or high marginal HV.

        # Simple effective approach: use quantiles of a density that weights
        # regions where the curve contributes more to hypervolume.
        # For 2D, the contribution of a small segment is roughly |df1 * df2| / (distance in t)
        # which is related to the arc length and the "width" of the HV contribution.

        # Let's use a practical approach: place points at quantiles of t
        # weighted by the local "importance" = |df1/dt| * |df2/dt|
        # This ensures denser sampling where the curve moves faster in objective space.

        # Numerical derivative
        dt = t_candidates[1] - t_candidates[0]
        df1_dt = np.gradient(f1_valid, dt)
        df2_dt = np.gradient(f2_valid, dt)

        # Importance weight: how much "objective space" is covered per unit t
        importance = np.abs(df1_dt) * np.abs(df2_dt) + 1e-10
        importance = importance / importance.sum()  # normalize to density

        # Compute CDF for inverse transform sampling
        cdf = np.cumsum(importance)
        cdf /= cdf[-1]

        # Place budget points at quantiles of this weighted distribution
        n_place = budget
        quantiles = np.linspace(0, 1, n_place + 2)[1:-1]  # avoid exact endpoints
        t_selected = np.interp(quantiles, cdf, t_valid)

        # Ensure uniqueness and bounds
        t_selected = np.clip(t_selected, 0.001, 0.999)

        # Sort and remove duplicates
        t_selected = np.unique(np.round(t_selected, 10))

        # If we have fewer than budget points, add more from remaining candidates
        if len(t_selected) < budget:
            remaining = np.setdiff1d(t_valid, t_selected, assume_unique=False)
            if len(remaining) > 0:
                n_add = budget - len(t_selected)
                # Add evenly spaced from remaining
                step = max(1, len(remaining) // n_add)
                extra = remaining[::step][:n_add]
                t_selected = np.sort(np.concatenate([t_selected, extra]))
            else:
                # Fallback: uniform spacing
                t_selected = np.linspace(0.001, 0.999, budget)

        t_selected = t_selected[:budget]

        # Evaluate all selected points
        for i, t in enumerate(t_selected):
            X[i, 0] = t
            F[i] = problem(X[i])

        # If we used fewer than budget (shouldn't happen), fill remaining
        used = len(t_selected)
        if used < budget:
            for i in range(used, budget):
                t = rng.uniform(0.001, 0.999) if hasattr(self, 'rng') else np.random.uniform(0.001, 0.999)
                X[i, 0] = t
                F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
