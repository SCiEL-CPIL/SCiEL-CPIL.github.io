"""Coordinate-descent optimizer for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coordinate descent with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(x)

        # Start near center (minimizer is in [-4,4]^5, center is 0)
        x = np.zeros(self.dim)
        fx = wrapped(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        steps = np.full(self.dim, 2.0)
        # Moving average of successful step magnitudes per dimension
        avg_steps = np.full(self.dim, 2.0)

        # Coordinate descent: cycle through dimensions
        # Each full cycle over all dims costs `dim` evaluations
        # Budget is 500, so we can do ~100 full cycles

        dim = self.dim
        call_count = 1  # initial evaluation

        # Phase 1: Coarse coordinate descent with large steps
        # Phase 2: Fine coordinate descent with small steps
        # Phase 3: Even finer for polish

        # We'll do multiple passes with decreasing step sizes
        # Each pass: for each dim, try +/- step, keep improvement

        # Strategy: 
        # - For each dimension, try moving in +step and -step directions
        # - If improvement, take it and grow step slightly
        # - If no improvement, shrink step
        # - This is a 1-D line search per coordinate

        # To maximize early progress (AOCC rewards early improvement):
        # Start with large steps to find the basin quickly, then refine.

        # Total budget: 500 calls
        # Use all of them in coordinate descent

        remaining = self.budget - call_count  # 499 remaining

        # We'll do a series of passes with geometrically decreasing base steps
        # Pass structure: for each base_step, do full coordinate sweeps
        # base_steps: 4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001
        # Each sweep over dim dims costs dim calls
        # 12 passes * 5 dims = 60 calls minimum, can do more sub-steps per dim

        # Better: for each dimension in a sweep, do a small 1-D search
        # Let's do: for each pass, for each dim, try a few steps

        # Simpler robust approach:
        # For each of the 499 remaining calls, pick a dimension cyclically,
        # try step in that dimension, accept if better, adapt step size.

        # Adaptive coordinate descent:
        # - Maintain step[d] for each dimension d
        # - For each call: pick dim = call_idx % dim
        # - Try x[dim] + step[dim] and x[dim] - step[dim] (2 calls per dim visit)
        # - Wait, that's 2 calls per dim, so 10 calls per full cycle
        # - 499 / 10 = ~50 full cycles

        # Let me restructure: 
        # Each "iteration" visits one dimension and uses 2 evaluations (try + and -)
        # Or use 1 evaluation: try a direction, if good keep and increase step, else decrease

        # 1-eval-per-dim approach (like the original ES but per-coordinate):
        # For dim d:
        #   y = x; y[d] += step[d] * direction
        #   if better: x = y; step[d] *= 1.2
        #   else: step[d] *= 0.7; flip direction

        # This uses 1 call per dim visit, so 5 calls per full cycle, 99 full cycles

        # Let's use this approach for efficiency

        directions = np.ones(dim)  # +1 or -1 for each dim
        steps = np.full(dim, 3.0)  # start with large steps

        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            d = i % dim
            y = x.copy()
            y[d] = np.clip(y[d] + directions[d] * steps[d], -5.0, 5.0)
            fy = wrapped(y)

            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Successful step: grow step slightly, keep direction
                steps[d] = min(steps[d] * 1.3, 10.0)
                # Update moving average
                avg_steps[d] = 0.8 * avg_steps[d] + 0.2 * steps[d]
            else:
                # Unsuccessful: shrink step, flip direction
                steps[d] = max(steps[d] * 0.6, 1e-6)
                directions[d] *= -1
                # Update moving average toward smaller
                avg_steps[d] = 0.8 * avg_steps[d] + 0.2 * steps[d]

        return best_x, best_fx
# EVOLVE-BLOCK-END
