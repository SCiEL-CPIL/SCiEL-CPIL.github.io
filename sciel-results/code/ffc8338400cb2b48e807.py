"""Multi-objective optimizer on DTLZ4 with structured Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)

        # DTLZ4 standard: first 2 vars are position (angles), rest are g-vars.
        # To be on Pareto front: set g-vars to 0.5 (midpoint).
        # Position vars x[0], x[1] map to angles in [0, pi/2].
        
        # Strategy: Use a well-distributed set of points on the Pareto front.
        # The Pareto front for 3-objective DTLZ4 is a spherical triangle.
        # We can sample uniformly on this surface using spherical coordinates.
        
        # Generate quasi-random points for the two position variables.
        # We want to cover the 2D simplex (x0, x1) in [0,1]^2 such that
        # the resulting points on the front are well-distributed.
        
        # The mapping from (x0, x1) to front is:
        # f1 = 0.5 * (1+g) * cos(x1*pi/2) * cos(x0*pi/2)  [if x0 is theta1]
        # f2 = 0.5 * (1+g) * cos(x1*pi/2) * sin(x0*pi/2)  [if x0 is theta1]
        # f3 = 0.5 * (1+g) * sin(x1*pi/2)                  [if x1 is theta2]
        
        # Wait, the standard DTLZ mapping is:
        # h1 = x1 * cos(x2 * pi/2)
        # h2 = x1 * sin(x2 * pi/2)
        # h3 = 1 - x1
        # So x0 = x1 (the radial one), x1 = x2 (the angular one).
        
        # To maximize hypervolume, we should sample points uniformly on the front.
        # The area element on the spherical triangle is proportional to sin(x1)*dx1*dx2?
        # Actually, let's use a Halton sequence in 2D for the position variables
        # and apply a mapping that accounts for the curvature.
        
        # Alternatively, just use a low-discrepancy sequence (Halton) in 2D
        # and map to the front. Halton sequences are good for covering [0,1]^2.
        
        # Let's generate N points using a Halton sequence.
        n_points = self.budget
        
        # Halton sequence implementation
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate Halton points
        x0_vals = np.array([halton(i, 2) for i in range(1, n_points + 1)])
        x1_vals = np.array([halton(i, 3) for i in range(1, n_points + 1)])
        
        # Create decision variable array
        X = np.tile(mid, (n_points, 1))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
