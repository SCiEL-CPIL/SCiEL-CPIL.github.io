"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer: coarse grid scan for basin identification,
    then local CMA-ES-style refinement in the best basin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid scan to find the global basin
        # Rastrigin has periodicity of 2π ≈ 6.28, but the box is [-5,5]
        # The shifted minimizer is in [-4,4]^5
        # Use a grid with spacing ~2.0 to cover the box with enough resolution
        # to identify which basin (integer offset) contains the minimum

        # For 5D, a full grid is expensive. Use stratified random sampling
        # to identify the best region, then refine.

        n_scan = 100  # number of evaluation points for scanning
        # Stratified sampling: divide each dimension into cells
        # 5^4 = 625 cells if we use 5 per dim, but that's too many
        # Use random stratified: 10 per dim = 100k cells, pick n_scan from them

        # Actually, let's use a smarter approach:
        # Sample at integer-like positions (near integers) since Rastrigin minima are at integers
        # The shift is hidden, so the true minimum is at some shifted integer position
        # Sample near integers in each dimension

        # Generate candidate centers near integers
        int_positions = np.arange(-4, 5, dtype=float)  # -4 to 4
        n_int = len(int_positions)  # 9

        # For each dimension, the minimum could be near any of these integers
        # Sample a subset of the full 9^5 grid
        n_candidates = min(500, 9 ** dim)
        
        if n_candidates <= 9 ** dim:
            # Can sample all or a subset
            if n_candidates == 9 ** dim:
                # Full grid
                grid = np.array(np.meshgrid(*([int_positions] * dim))).T.reshape(-1, dim)
            else:
                # Random subset
                idx = rng.choice(9 ** dim, size=n_candidates, replace=False)
                grid = np.zeros((n_candidates, dim))
                for d in range(dim):
                    grid[:, d] = int_positions[idx % 9]
                    idx //= 9
        else:
            grid = rng.uniform(-4.5, 4.5, size=(n_candidates, dim))

        # Evaluate grid points
        best_fx = np.inf
        best_x = np.zeros(dim)
        scan_calls = 0

        for i in range(len(grid)):
            if time.time() > deadline - 60:
                break
            x = grid[i]
            fx = func(x)
            scan_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Local refinement using (1+1)-ES around the best basin
        remaining = budget - scan_calls
        if remaining <= 0:
            return best_x, best_fx

        x = best_x.copy()
        fx = best_fx
        sigma = 0.5  # Start with smaller step since we're already in a basin

        # Adaptive step size with success rule
        successes = 0
        n_window = 10
        window = []

        for i in range(remaining):
            if time.time() > deadline - 10:
                break

            # Occasionally add a larger jump to escape local minima
            if rng.random() < 0.05:
                step_scale = sigma * 3.0
            else:
                step_scale = sigma

            y = np.clip(x + step_scale * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                successes += 1
                window.append(1)
                sigma *= 1.3
            else:
                window.append(0)
                sigma *= 0.9

            # Adjust sigma based on recent success rate
            if len(window) >= n_window:
                recent_success = sum(window[-n_window:]) / n_window
                if recent_success > 0.7:
                    sigma *= 1.1
                elif recent_success < 0.3:
                    sigma *= 0.9
                window = window[-n_window:]

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
