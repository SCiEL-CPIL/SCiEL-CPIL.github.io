"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: random exploration, adaptive (1+1)-ES with restarts, and fine-tuning."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        def time_up():
            return time.time() >= deadline

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random exploration to find a good starting region
        # Use about 20% of budget for random sampling
        n_random = max(10, budget // 5)
        best_x = None
        best_f = np.inf

        for i in range(n_random):
            if time_up():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try some structured points
        structured_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        for sp in structured_points:
            if time_up():
                break
            fx = func(sp)
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        # Phase 2: Adaptive (1+1)-ES with restarts
        # Use remaining budget
        remaining = budget - n_random - len(structured_points)
        if remaining < 10:
            remaining = max(10, budget - n_random)

        sigma = 2.0
        x = best_x.copy()
        fx = best_f
        no_improve = 0
        max_no_improve = 30  # Restart if no improvement for this many steps
        restart_count = 0
        max_restarts = 5

        for i in range(remaining):
            if time_up():
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                best_x, best_f = x.copy(), fx
                sigma *= 1.3
                no_improve = 0
            else:
                sigma *= 0.85
                no_improve += 1

            # Check for restart
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                no_improve = 0
                sigma = max(0.5, sigma * 0.5)
                # Try a few random restarts
                for _ in range(3):
                    if time_up():
                        break
                    xr = x + sigma * rng.standard_normal(dim)
                    xr = np.clip(xr, -5, 5)
                    fr = func(xr)
                    if fr < fx:
                        x, fx = xr, fr
                        best_x, best_f = x.copy(), fx
                        no_improve = 0
                        break

            # Ensure sigma doesn't get too small
            if sigma < 0.01:
                sigma = 0.01

        # Phase 3: Fine-tuning with very small steps
        if not time_up() and best_x is not None:
            fine_budget = min(50, max(10, budget - n_random - remaining))
            sigma_fine = 0.1
            x = best_x.copy()
            fx = best_f

            for i in range(fine_budget):
                if time_up():
                    break
                y = x + sigma_fine * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx
                    sigma_fine *= 1.2
                else:
                    sigma_fine *= 0.8

                if sigma_fine < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
