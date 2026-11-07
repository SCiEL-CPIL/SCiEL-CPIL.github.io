"""Multi-start adaptive (1+1)-ES with restarts and polish phase for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with restarts and polish phase for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Deterministic diverse starts for early basin discovery
        starts = [np.zeros(dim)]
        for scale in [1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        for _ in range(20):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: Main search with adaptive (1+1)-ES + restarts (~70% of remaining budget)
        remaining = budget - n_calls[0]
        main_limit = int(n_calls[0] + remaining * 0.7)

        sigma = 2.0
        success_window = []
        window_size = 15
        restart_threshold = 60
        max_restarts = 5
        restart_count = 0

        while n_calls[0] < main_limit and time.time() < deadline:
            no_improve = 0

            while n_calls[0] < main_limit and time.time() < deadline and no_improve <= restart_threshold:
                candidate = best_x + sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    success_window.append(1)
                    no_improve = 0
                else:
                    success_window.append(0)
                    no_improve += 1

                # Adapt sigma using 1/5 success rule
                if len(success_window) >= window_size:
                    sr = sum(success_window) / window_size
                    if sr > 0.3:
                        sigma *= 1.25
                    elif sr < 0.2:
                        sigma *= 0.8
                    success_window = success_window[-window_size // 2:]

                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)

            # If we exited due to no_improve threshold, restart from a new point
            if n_calls[0] < main_limit and time.time() < deadline and no_improve > restart_threshold:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                start_x = rng.uniform(-4.5, 4.5, dim)
                fx = safe_func(start_x)
                if fx < best_f:
                    best_f = fx
                    best_x = start_x.copy()
                sigma = rng.uniform(1.0, 3.0)
                success_window = []

        # Phase 3: High-precision polish with smaller sigma and aggressive reduction
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.1, 1e-4)
            polish_window = []
            polish_window_size = 5

            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.7
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    sr = sum(polish_window) / polish_window_size
                    if sr > 0.4:
                        polish_sigma *= 1.3
                    elif sr < 0.15:
                        polish_sigma *= 0.75
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-10)
                polish_sigma = min(polish_sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
