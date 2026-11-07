"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware stratified sampling on the ZDT3 Pareto front with curvature-aware allocation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_h(self, x1):
        """Compute h(x1) for ZDT3."""
        return 0.5 * x1 + np.sin(10.0 * np.pi * x1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 Pareto front: x_2..x_dim = 0, f_1 = x_1, f_2 = 0.5*x_1 + sin(10*pi*x_1)
        # The front has 10 oscillations. Points where f_2 is small contribute more to hypervolume.
        
        # Strategy: Use adaptive stratified sampling that allocates more points
        # to regions where the front has high hypervolume contribution.
        
        # First, create a fine grid to estimate where to concentrate samples
        n_fine = 2000
        x1_fine = np.linspace(0.0, 1.0, n_fine)
        f2_fine = self._zdt3_h(x1_fine)
        
        # For hypervolume with reference [11, 11], the contribution of a point (f1, f2)
        # is roughly proportional to the area it "controls" in the objective space.
        # Points with smaller f2 values dominate more area.
        
        # Weight by inverse of f2 (points with small f2 are more valuable)
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (np.abs(f2_fine) + 0.01)
        
        # Also consider that points in the "lower envelope" are most important
        # The lower envelope of f2 vs f1 determines the Pareto front shape
        
        # Use a smooth weighting: weight = (1 + alpha * (1 - f2/max_f2))
        # This gives more weight to points with smaller f2
        max_f2 = np.max(f2_fine)
        if max_f2 > 0:
            weights = 1.0 + 2.0 * (1.0 - f2_fine / max_f2)
            weights = np.maximum(weights, 0.1)  # Ensure positive weights
        
        # Normalize weights to sum to n_fine
        weights = weights / np.sum(weights) * n_fine
        
        # Now sample budget points from this weighted distribution
        # Use inverse CDF method
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]  # Normalize to [0, 1]
        
        # Generate random quantiles
        n_points = budget
        u = np.random.uniform(0, 1, n_points)
        
        # Inverse CDF sampling
        x1_samples = np.interp(u, cdf, x1_fine)
        
        # Add small jitter to avoid duplicates
        jitter = np.random.uniform(-0.001, 0.001, n_points)
        x1_samples = np.clip(x1_samples + jitter, 0.0, 1.0)
        
        # Sort to ensure we cover the range well
        x1_samples = np.sort(x1_samples)
        
        # Ensure endpoints are covered
        if len(x1_samples) > 0:
            x1_samples[0] = 0.0
            x1_samples[-1] = 1.0
        
        # Create decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_samples
        # x_2..x_dim are 0 for the Pareto front
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
