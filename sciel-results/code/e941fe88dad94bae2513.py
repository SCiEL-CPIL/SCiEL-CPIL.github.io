"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: parallel CMA-ES-like search with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 2.0

        def calls_left():
            return budget - calls

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma (budget: ~40%)
        n_starts = 3
        per_start = (budget * 40) // 100 // n_starts
        starts = [np.zeros(dim), rng.uniform(-4, 4, dim), rng.uniform(-3, 3, dim)]

        for s in starts:
            if calls >= budget or not time_left():
                break
            x = s.copy()
            fx = func(x)
            calls += 1
            update_best(x, fx)
            sigma = 2.5
            for _ in range(per_start):
                if calls >= budget or not time_left():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.25, 4.0)
                else:
                    sigma = max(sigma * 0.85, 0.05)

        # Phase 2: Focused (1+1)-ES from best found (budget: ~30%)
        if calls < budget and time_left():
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            steps2 = min(budget * 30 // 100, budget - calls)
            for _ in range(steps2):
                if calls >= budget or not time_left():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 2.0)
                else:
                    sigma = max(sigma * 0.9, 0.01)

        # Phase 3: Coordinate-wise refinement (budget: ~20%)
        if calls < budget and time_left():
            x4 = best_x.copy()
            fx4 = best_f
            remaining = budget - calls
            for coord in range(dim):
                if calls >= budget or not time_left():
                    break
                for direction in [1.0, -1.0]:
                    if calls >= budget or not time_left():
                        break
                    step = 1.0
                    for _ in range(20):
                        if calls >= budget or not time_left():
                            break
                        trial_x = x4.copy()
                        trial_x[coord] = np.clip(x4[coord] + direction * step, -5, 5)
                        trial_f = func(trial_x)
                        calls += 1
                        if trial_f < fx4:
                            x4, fx4 = trial_x, trial_f
                            update_best(x4, fx4)
                            step *= 1.5
                        else:
                            step *= 0.5
                            if step < 0.01:
                                break

        # Phase 4: Final precision polish - small random perturbations + coordinate scan
        if calls < budget and time_left():
            x5 = best_x.copy()
            fx5 = best_f
            # Fine random search
            sigma5 = 0.1
            steps5 = min(budget - calls, 50)
            for _ in range(steps5):
                if calls >= budget or not time_left():
                    break
                y = np.clip(x5 + sigma5 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx5:
                    x5, fx5 = y, fy
                    sigma5 = min(sigma5 * 1.2, 0.5)
                else:
                    sigma5 = max(sigma5 * 0.8, 0.005)

            # Final coordinate scan with tight steps
            if calls < budget and time_left():
                x6 = best_x.copy()
                fx6 = best_f
                for coord in range(dim):
                    if calls >= budget or not time_left():
                        break
                    for direction in [1.0, -1.0]:
                        if calls >= budget or not time_left():
                            break
                        step = 0.1
                        for _ in range(10):
                            if calls >= budget or not time_left():
                                break
                            trial_x = x6.copy()
                            trial_x[coord] = np.clip(x6[coord] + direction * step, -5, 5)
                            trial_f = func(trial_x)
                            calls += 1
                            if trial_f < fx6:
                                x6, fx6 = trial_x, trial_f
                                update_best(x6, fx6)
                                step *= 1.3
                            else:
                                step *= 0.6
                                if step < 0.001:
                                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
