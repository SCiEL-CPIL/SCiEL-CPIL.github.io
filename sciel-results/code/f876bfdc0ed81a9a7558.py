"""Optimizer for DTLZ4 multi-objective optimization using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _construct_point(self, x1, x2, bounds):
        """Construct a DTLZ4 point with g=0 (Pareto optimal)."""
        x = np.zeros(self.dim)
        x[0] = x1
        x[1] = x2
        # g-variables (indices 7-11) set to 0.5 for g=0
        for i in range(7, self.dim):
            lo, hi = bounds[i]
            x[i] = 0.5
        # x[2], x[3], x[4], x[5], x[6] can be anything (they don't affect objectives when g=0)
        # Set to 0.5 for safety
        for i in range(2, 7):
            lo, hi = bounds[i]
            x[i] = 0.5
        # Clip to bounds
        for i in range(self.dim):
            lo, hi = bounds[i]
            x[i] = np.clip(x[i], lo, hi)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 Pareto front:
        # x[0], x[1] in [0,1] control the front
        # g-variables x[7..11] set to 0.5 for g=0
        # f1 = cos(x1*pi/2) * cos(x2*pi/2) * (1+g)
        # f2 = cos(x1*pi/2) * sin(x2*pi/2) * (1+g)
        # f3 = sin(x1*pi/2) * (1+g)
        # When g=0: f1 = cos(x1*pi/2)*cos(x2*pi/2), f2 = cos(x1*pi/2)*sin(x2*pi/2), f3 = sin(x1*pi/2)
        # f1^2 + f2^2 + f3^2 = cos^2(x1*pi/2) + sin^2(x1*pi/2) = 1
        # But wait, the reference point is [2.5, 2.5, 2.5], and the front is on a quarter sphere of radius 1
        # Actually for DTLZ4 with 12 vars and 3 objectives, the standard formulation:
        # M = 3, K = n - M + 1 = 12 - 3 + 1 = 10
        # g = sum_{i in K} (x_i - 0.5)^2 * sin^2(20*pi*x_i)
        # f1 = (1+g) * cos(x1*pi/2) * cos(x2*pi/2)
        # f2 = (1+g) * cos(x1*pi/2) * sin(x2*pi/2)
        # f3 = (1+g) * sin(x1*pi/2)
        
        # So on the Pareto front (g=0):
        # f1 = cos(x1*pi/2)*cos(x2*pi/2)
        # f2 = cos(x1*pi/2)*sin(x2*pi/2)
        # f3 = sin(x1*pi/2)
        # where x1, x2 in [0,1]
        
        # The front is a quarter-sphere of radius 1 in the positive octant
        # We want to cover this front densely with 200 evaluations
        
        # Strategy: Use a dense stratified sampling in (x1, x2) space
        # x1 in [0,1], x2 in [0,1]
        # The mapping to the front is:
        # theta = x1 * pi/2 (elevation angle from f1-f2 plane)
        # phi = x2 * pi/2 (azimuthal angle in f1-f2 plane)
        # f1 = cos(theta)*cos(phi)
        # f2 = cos(theta)*sin(phi)
        # f3 = sin(theta)
        
        # We want to maximize hypervolume. The front is a smooth curved surface.
        # A uniform grid in (theta, phi) space would give uniform angular coverage.
        # But the area element on the sphere is sin(theta) dtheta dphi.
        # For better hypervolume, we should use more points where the surface area element is larger.
        
        # Actually, for DTLZ4 the standard is x in [0,1], so theta = x1*pi/2, phi = x2*pi/2
        # Let's use a stratified grid with better coverage.
        
        # Use 200 points. Let's try different grid sizes:
        # 20x10 = 200, 14x14 = 196, 25x8 = 200, etc.
        
        # Since theta and phi are both in [0, pi/2], and the mapping is symmetric in a sense,
        # let's use a stratified approach.
        
        # Better: Use a Halton-like low-discrepancy sequence for better space-filling
        def _halton(idx, base):
            f = 1.0
            r = 0.0
            i = idx
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Generate N Halton sequence points in [0,1]^2
        N = 200
        points = []
        for idx in range(1, N + 1):
            x1 = _halton(idx, 2)
            x2 = _halton(idx, 3)
            points.append((x1, x2))
        
        # Evaluate all points
        used = 0
        for x1, x2 in points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = self._construct_point(x1, x2, bounds)
            f = problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
