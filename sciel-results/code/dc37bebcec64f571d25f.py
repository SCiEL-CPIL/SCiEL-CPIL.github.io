"""Rewritten optimizer for BBOB f13 (Sharp ridge) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer with multi-start and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def timed_call(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            return f

        # --- Phase 1: Initial random sampling ---
        n_init = min(30, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = timed_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: Adaptive population-based search ---
        # Population size
        lam = 10
        # Initial step size
        sigma = 2.0
        # Track best
        x_best = best_x.copy()
        f_best = best_f

        # Success rate based step size adaptation
        # Keep a history of successes
        success_history = []
        success_window = 10

        while calls < budget:
            # Check deadline
            if time.time() > deadline - 10.0:
                break

            # Generate population
            candidates = []
            for i in range(lam):
                # Mix of exploration and exploitation
                if i == 0:
                    # Best point + small perturbation
                    y = np.clip(x_best + sigma * 0.2 * rng.standard_normal(dim), -5, 5)
                elif i < lam // 3:
                    # Medium step from best
                    y = np.clip(x_best + sigma * rng.standard_normal(dim), -5, 5)
                elif i < 2 * lam // 3:
                    # Larger step
                    y = np.clip(x_best + sigma * 1.5 * rng.standard_normal(dim), -5, 5)
                else:
                    # Random restart component
                    y = rng.uniform(-5, 5, dim)

                fy = timed_call(y)
                candidates.append((fy, y))

                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()

            # Sort candidates
            candidates.sort(key=lambda t: t[0])

            # Success rate: count how many of the top half are better than the previous best
            top_half = candidates[:lam // 2]
            successes = sum(1 for fy, _ in top_half if fy < f_best + 1e-10)
            success_rate = successes / (lam // 2)

            # Adapt sigma
            if success_rate > 0.5:
                sigma *= 1.2
            elif success_rate < 0.3:
                sigma *= 0.7
            else:
                sigma *= 0.95

            sigma = max(0.001, min(sigma, 5.0))

            # Update center: weighted recombination of top candidates
            weights = np.exp(-np.arange(lam) * 0.3)
            weights /= np.sum(weights)
            new_center = np.zeros(dim)
            for w, (_, y) in zip(weights, candidates):
                new_center += w * y
            # Blend with current best
            x_best = 0.7 * x_best + 0.3 * new_center

        # --- Phase 3: Local refinement with (1+1)-ES ---
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 5.0:
            local_sigma = max(sigma * 0.1, 0.01)
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                y = np.clip(x_best + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_call(y)
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    local_sigma *= 1.3
                else:
                    local_sigma *= 0.85

                local_sigma = max(0.0001, min(local_sigma, 1.0))

        return x_best, f_best
# EVOLVE-BLOCK-END
