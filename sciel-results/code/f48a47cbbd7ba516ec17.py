"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-lite with adaptive step sizes and local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        total_budget = self.budget
        calls = [0]

        def safe_func(x):
            if calls[0] >= total_budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # Use a combination of random points and structured points
        n_starts = min(10, total_budget // 50)  # at least 1, at most 10
        starts = []

        # Center point
        starts.append(np.zeros(dim))

        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        # Add some structured points for Rastrigin (which has periodic minima)
        if n_starts > 2:
            for k in range(min(4, n_starts - 2)):
                point = rng.uniform(-4, 4, dim)
                # Snap some coordinates to integers (Rastrigin minima are near integers)
                for d in range(dim):
                    if rng.random() < 0.5:
                        point[d] = np.round(point[d])
                starts.append(point)

        # Keep track of best found so far
        best_x = None
        best_f = np.inf

        # Phase 2: For each start, run a local optimizer
        # Use adaptive step size per dimension (diagonal CMA-ES)
        budget_per_start = max(10, (total_budget - 1) // len(starts))

        for i, x0 in enumerate(starts):
            if calls[0] >= total_budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Local optimization from x0
            x = x0.copy()
            fx = safe_func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive step sizes per dimension
            sigma = np.ones(dim) * 1.0
            success_count = 0
            total_steps = 0
            max_local_steps = budget_per_start

            for step in range(max_local_steps):
                if calls[0] >= total_budget:
                    break
                if time.time() > deadline - 2.0:
                    break

                # Generate perturbation with adaptive step sizes
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = safe_func(y)

                total_steps += 1

                if fy < fx:
                    # Success: move to y, increase step sizes
                    x, fx = y, fy
                    sigma *= 1.3
                    success_count += 1

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease step sizes
                    sigma *= 0.8

                # Adapt step sizes based on success rate over recent steps
                if total_steps % 10 == 0:
                    success_rate = success_count / 10
                    success_count = 0
                    if success_rate > 0.6:
                        sigma *= 1.1
                    elif success_rate < 0.3:
                        sigma *= 0.9

                    # Clamp step sizes
                    sigma = np.clip(sigma, 0.01, 5.0)

            # Local polishing: try to refine with smaller steps
            if calls[0] < total_budget and time.time() < deadline - 2.0:
                polish_steps = min(20, total_budget - calls[0])
                for _ in range(polish_steps):
                    if calls[0] >= total_budget:
                        break
                    if time.time() > deadline - 1.0:
                        break
                    direction = rng.standard_normal(dim)
                    y = np.clip(x + 0.1 * sigma * direction, -5, 5)
                    fy = safe_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Phase 3: If we have budget left, do a final refinement around best_x
        if best_x is not None and calls[0] < total_budget and time.time() < deadline - 1.0:
            x = best_x.copy()
            fx = best_f
            sigma = np.ones(dim) * 0.5
            remaining = total_budget - calls[0]
            max_refine = min(remaining, 50)

            for _ in range(max_refine):
                if calls[0] >= total_budget:
                    break
                if time.time() > deadline - 0.5:
                    break
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                sigma = np.clip(sigma, 0.005, 3.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
