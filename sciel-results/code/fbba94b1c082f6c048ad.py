"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase (1+1)-ES with restarts for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        # Phase 1: Coarse random exploration to find a good starting basin
        best_x = None
        best_f = np.inf
        n_init = 30
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Main optimization with (1+1)-ES + adaptive restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        no_improve = 0
        max_no_improve = 40  # restart threshold

        while calls[0] < budget and time.time() < deadline:
            # Generate trial
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = call_func(y)

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.1
                no_improve = 0
            else:
                sigma *= 0.95
                no_improve += 1

            # Restart logic: if stuck, kick from global best with larger perturbation
            if no_improve >= max_no_improve:
                no_improve = 0
                sigma = 1.0
                # Jump to a point near the global best with moderate perturbation
                x = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
                fx = call_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Also try a few exploratory steps
                for _ in range(5):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = call_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                # Reset sigma to a reasonable value
                sigma = 0.8
                max_no_improve = 30  # slightly more aggressive after first restart

            # If sigma gets too small, boost it
            if sigma < 0.05:
                sigma = 0.3

        # Phase 3: Fine-tuning near the best point with very small steps
        if calls[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for i in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
