"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Initialize at center of box
        x = np.zeros(d)
        fx = func(x)
        calls = 1

        # Adaptive step size
        sigma = 3.0
        # Track success rate for step size adaptation
        successes = 0
        window = 10
        success_count = 0

        # Momentum / direction tracking
        best_x = x.copy()
        best_fx = fx

        # Use a simple CMA-like approach with covariance adaptation
        # For 5D, we can use a simpler adaptive method

        # Track average of last few successful directions
        recent_dirs = []

        for i in range(budget - 1):
            if not time_ok():
                break

            # Generate candidate
            direction = rng.standard_normal(d)
            # Add some momentum from recent successful directions
            if recent_dirs and len(recent_dirs) > 0:
                avg_dir = np.mean(recent_dirs, axis=0)
                norm_avg = np.linalg.norm(avg_dir)
                if norm_avg > 1e-10:
                    avg_dir = avg_dir / norm_avg
                    direction = 0.7 * direction + 0.3 * avg_dir
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                fx = fy
                x = y.copy()
                success_count += 1
                if len(recent_dirs) >= window:
                    recent_dirs.pop(0)
                recent_dirs.append(direction)
                # Increase step size
                sigma *= 1.15
            else:
                # Failure
                # Decrease step size
                sigma *= 0.85

            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Adapt sigma based on success rate in window
            if len(recent_dirs) == window:
                # If we've been successful, increase sigma more
                pass

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
