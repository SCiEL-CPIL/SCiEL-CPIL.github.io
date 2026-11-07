"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution (global) + Nelder-Mead (local polish)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s / 330s kill

        calls = [0]

        def tracked_func(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Differential Evolution ---
        pop_size = 20
        F = 0.8      # scale factor
        CR = 0.9     # crossover rate

        # Initial population
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        fpop = np.array([tracked_func(x) for x in pop])

        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        gen = 0
        max_gen = (budget // pop_size) - 1  # leave calls for local polish

        for gen in range(max_gen):
            if time.time() > deadline:
                break
            if calls[0] >= budget:
                break

            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if calls[0] >= budget:
                    break

                # Select three distinct random indices
                while True:
                    r1 = rng.integers(0, pop_size)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(0, pop_size)
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = rng.integers(0, pop_size)
                    if r3 != i and r3 != r1 and r3 != r2:
                        break

                # Differential mutation
                v = pop[i] + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[i])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                jrand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                ft = tracked_func(trial)

                if ft <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

        # --- Phase 2: Nelder-Mead local polish from best point ---
        remaining = budget - calls[0]
        if remaining > 10 and time.time() < deadline:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': max(1, remaining // 3),
                             'xatol': 1e-8, 'fatol': 1e-10,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

        # If we still have budget and time, try a few more local restarts
        if calls[0] < budget and time.time() < deadline:
            for _ in range(min(3, (budget - calls[0]) // 10)):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Perturb best_x slightly
                x0 = best_x + rng.normal(0, 0.1, dim)
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(
                        tracked_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 20,
                                 'xatol': 1e-8, 'fatol': 1e-10,
                                 'adaptive': True}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
