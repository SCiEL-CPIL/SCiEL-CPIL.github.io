"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive DE (JADE-style) with local refinement, time-managed."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track calls
        calls = [0]

        def track_call(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Initialize population
        pop_size = 20
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([track_call(p) for p in pop])

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # JADE parameters
        c = 0.1  # crossover rate base
        F = 0.5  # differential weight base
        F_mean = 0.5
        c_mean = 0.1

        # Phase 1: DE until ~80% of budget used or time runs out
        phase1_budget = int(budget * 0.75)
        de_calls_used = pop_size

        while calls[0] < phase1_budget and time.time() < deadline:
            # Select base, r1, r2 (distinct from i)
            i = rng.integers(pop_size)
            others = [j for j in range(pop_size) if j != i]
            r1 = others[rng.integers(len(others))]
            r2 = others[rng.integers(len(others))]

            # Adaptive F and c
            F_i = rng.exponential(F_mean)
            c_i = rng.beta(1, 1) * 2 * c_mean

            # DE/rand/1 mutation
            v = pop[i] + F_i * (pop[r1] - pop[r2])
            v = np.clip(v, -5, 5)

            # Binomial crossover
            j_rand = rng.integers(dim)
            trial = pop[i].copy()
            for j in range(dim):
                if rng.random() < c_i or j == j_rand:
                    trial[j] = v[j]

            f_trial = track_call(trial)
            de_calls_used += 1

            if f_trial < fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                # Update adaptive parameters
                F_mean = 0.9 * F_mean + 0.1 * F_i
                c_mean = 0.9 * c_mean + 0.1 * c_i
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
            else:
                # Blend with current
                F_mean = 0.9 * F_mean + 0.1 * 0.5
                c_mean = 0.9 * c_mean + 0.1 * 0.1

        # Phase 2: Local refinement using (1+1)-ES from best point
        remaining_budget = budget - calls[0]
        if remaining_budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = track_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
