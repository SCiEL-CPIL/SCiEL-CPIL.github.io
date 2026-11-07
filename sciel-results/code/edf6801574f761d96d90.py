"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step sizes for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initialize at origin (common for BBOB shifted functions)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size per dimension
        sigma = np.full(dim, 1.0)
        success_counts = np.zeros(dim)
        trial_counts = np.zeros(dim)

        calls_made = 1

        # Parameters for adaptation
        success_target = 1/5  # 1/5 success rule
        step_up = 1.2
        step_down = 0.8

        # Try to find a better starting point with a few random samples
        # but keep it minimal to save budget
        for _ in range(min(10, budget - 1)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            calls_made += 1
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
                x = x_try.copy()
                fx = fx_try
                # Reset adaptation state when we jump to a new best
                sigma = np.full(dim, 1.0)
                success_counts = np.zeros(dim)
                trial_counts = np.zeros(dim)

        # Main optimization loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate a candidate step
            # Use a mixture of large and small steps to handle the bent cigar landscape
            # Bent cigar has one hard direction, so we need to be careful with step sizes
            direction = rng.standard_normal(dim)
            # Scale direction by current sigma
            step = sigma * direction
            x_new = np.clip(x + step, -5, 5)
            fx_new = func(x_new)
            calls_made += 1

            if fx_new < fx:
                # Success: move to new point
                x = x_new
                fx = fx_new
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                
                # Update adaptation: increase sigma for dimensions that helped
                # We need to figure out which dimensions contributed
                # Simple approach: increase all sigmas slightly
                for d in range(dim):
                    success_counts[d] += 1
                    trial_counts[d] += 1
            else:
                # Failure: keep x, decrease sigma
                for d in range(dim):
                    trial_counts[d] += 1

            # Adapt sigma periodically
            if calls_made % 10 == 0:
                for d in range(dim):
                    if trial_counts[d] > 0:
                        success_rate = success_counts[d] / trial_counts[d]
                        if success_rate > success_target:
                            sigma[d] *= step_up
                        elif success_rate < success_target:
                            sigma[d] *= step_down
                        # Clamp sigma
                        sigma[d] = np.clip(sigma[d], 0.01, 5.0)
                    # Reset counters
                    success_counts[d] = 0
                    trial_counts[d] = 0

            # Additional heuristic: if we're stuck (no improvement in last 50 calls),
            # try a larger random jump
            # This is a simple restart mechanism
            if calls_made % 50 == 0 and calls_made > 100:
                # Check if we've made progress recently
                # For simplicity, just occasionally try a larger step
                if rng.random() < 0.1:
                    x_jump = np.clip(x + rng.uniform(-2, 2, dim), -5, 5)
                    fx_jump = func(x_jump)
                    calls_made += 1
                    if fx_jump < fx:
                        x = x_jump
                        fx = fx_jump
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        sigma = np.full(dim, 1.0)
                        success_counts = np.zeros(dim)
                        trial_counts = np.zeros(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
