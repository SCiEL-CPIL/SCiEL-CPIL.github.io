"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES restarts with time guard for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        # We'll track the best solution found
        best_x = None
        best_f = np.inf

        # Strategy: multiple short (1+1)-ES runs with different starting points
        # and sigma schedules. The bent cigar function has one very narrow valley,
        # so we need good exploration of the first dimension while exploiting
        # the flat directions.

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Create a set of starting points: some random, some along axes
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))
        # Points along each axis (positive and negative)
        for d in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[d] = sign * 4.0
                starts.append(pt)
        # Some more structured points
        for _ in range(5):
            pt = rng.uniform(-5, 5, dim)
            # Make one coordinate large, others small
            d = rng.integers(0, dim)
            pt[d] = sign * rng.uniform(2, 5)
            starts.append(pt)

        # Remove duplicates approximately
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=0.1):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Run (1+1)-ES from each starting point
        calls_used = 0

        for i, x0 in enumerate(unique_starts):
            if time.time() > deadline - 5.0:
                break
            if calls_used >= budget - 1:
                break

            # Allocate remaining budget proportionally, but give more to early runs
            remaining = budget - 1 - calls_used
            n_runs_left = len(unique_starts) - i
            # Give each run at least 5 calls, up to remaining
            alloc = max(5, remaining // n_runs_left)
            alloc = min(alloc, remaining)

            x = x0.copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # For bent cigar, the first dimension is the hard one.
            # Use a sigma that allows good exploration.
            for _ in range(alloc):
                if time.time() > deadline - 2.0:
                    break
                if calls_used >= budget - 1:
                    break

                # Anisotropic perturbation: larger step in first dimension
                step = sigma * rng.standard_normal(dim)
                # For bent cigar, f(x) = (x0 + 2*x1)^2 + sum(x2^2)
                # The valley is along x0 = -2*x1, so we want to explore that direction
                # Use a transformation that aligns with the valley structure
                # Actually, let's just use standard ES but with careful sigma control

                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)

        # Final local refinement around best point
        if best_x is not None and time.time() < deadline - 10.0 and calls_used < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                if calls_used >= budget - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
