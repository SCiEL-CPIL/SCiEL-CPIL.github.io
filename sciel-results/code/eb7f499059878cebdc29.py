"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and lightweight restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Main (1+1)-ES with adaptive restarts
        polish_budget = max(30, budget // 20)
        search_budget = budget - polish_budget

        # Start at origin
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 5.0
        no_improve = 0
        restart_count = 0
        max_restarts = 8

        while total_calls < search_budget and not check_deadline():
            # Adaptive stall threshold: shorter as we use more budget
            progress = total_calls / search_budget
            max_no_improve = max(20, int(60 * (1.0 - progress)))

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma = min(sigma_max, sigma * 1.3)
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                sigma = max(sigma_min, sigma * 0.85)
                no_improve += 1

            # Restart: if stalled, perturb from best point or try new random
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                if restart_count <= 3 and best_x is not None:
                    # First few restarts: perturb near best
                    x = np.clip(best_x + 1.5 * rng.standard_normal(d), -5, 5)
                else:
                    # Later restarts: more random exploration
                    x = rng.uniform(-4.5, 4.5, d)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                sigma = 1.0
                no_improve = 0

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2

            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(0.5, sigma * 1.15)
                else:
                    sigma = max(1e-8, sigma * 0.88)

                if sigma < 1e-8:
                    break

        # Phase 3: Coordinate-wise fine polish
        if best_x is not None and not check_deadline():
            remaining = budget - total_calls
            if remaining > 5:
                x = best_x.copy()
                fx = best_f
                step_sizes = [0.05, 0.01, 0.001]
                for ss in step_sizes:
                    for _ in range(min(5, remaining)):
                        if check_deadline():
                            break
                        for i in range(d):
                            if check_deadline():
                                break
                            for sign in [1, -1]:
                                if check_deadline():
                                    break
                                y = x.copy()
                                y[i] = np.clip(y[i] + sign * ss, -5, 5)
                                fy = func(y)
                                total_calls += 1
                                if fy < fx:
                                    x, fx = y, fy
                                    if fx < best_f:
                                        best_f, best_x = fx, x.copy()
                                if total_calls >= budget:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
