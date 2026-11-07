"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES variants with diverse restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 300s wall-clock

        calls = 0
        best_f = np.inf
        best_x = None

        def tracked_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= self.budget:
                raise TimeoutError()
            calls += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse global exploration with multiple random starts
        # and adaptive (1+1)-ES
        num_restarts = 8
        calls_per_restart = self.budget // (num_restarts + 1)
        remaining_for_local = self.budget // 4

        # First, do a broad random search to find promising regions
        for i in range(min(50, self.budget)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = tracked_func(x)

        # Phase 2: Multiple (1+1)-ES runs from best found points
        # Sort by fitness and pick diverse starting points
        # We'll run several (1+1)-ES variants with different step sizes

        # Collect some candidate starting points
        candidates = []
        # Add the best point found so far
        if best_x is not None:
            candidates.append(best_x.copy())

        # Add a few more random points spread across the domain
        for _ in range(4):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            f = tracked_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            candidates.append(x.copy())

        # Run (1+1)-ES from each candidate
        for start_idx, start_x in enumerate(candidates):
            if time.time() > deadline:
                break
            if calls >= self.budget - remaining_for_local:
                break

            x = start_x.copy()
            fx = tracked_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma
            sigma = 0.8 + 0.3 * start_idx  # Different step sizes for diversity
            success_count = 0
            total_count = 0
            stall_counter = 0

            while calls < self.budget - remaining_for_local and time.time() < deadline:
                # Adapt sigma
                if total_count > 5:
                    success_rate = success_count / total_count
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.85
                    if total_count > 15:
                        success_count = 0
                        total_count = 0

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    stall_counter = 0
                else:
                    stall_counter += 1
                    # If stalled, try a larger jump
                    if stall_counter > 10:
                        y2 = np.clip(x + sigma * 3 * rng.standard_normal(self.dim), -5, 5)
                        fy2 = tracked_func(y2)
                        total_count += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma *= 0.9
                            stall_counter = 0
                        else:
                            # Restart from a random point
                            x = rng.uniform(-4, 4, self.dim)
                            fx = tracked_func(x)
                            sigma = 1.0
                            stall_counter = 0

        # Phase 3: Local refinement around the best point
        if best_x is not None and time.time() < deadline - 5:
            x = best_x.copy()
            fx = tracked_func(x)
            sigma = 0.3
            success_count = 0
            total_count = 0

            while calls < self.budget and time.time() < deadline:
                if total_count > 5:
                    success_rate = success_count / total_count
                    if success_rate > 0.3:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    if total_count > 10:
                        success_count = 0
                        total_count = 0

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                else:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
