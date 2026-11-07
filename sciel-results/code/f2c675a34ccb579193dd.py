"""Differential Evolution + Nelder-Mead polish for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Population-based DE/rand/1 with adaptive F and final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s wall-clock, 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(20, max(8, budget // 15))

        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])

        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Adaptive mutation factor
        F = 0.8
        CR = 0.5

        # Count evaluations used
        evals_used = pop_size

        # DE loop
        generation = 0
        stagnation_count = 0
        prev_best_f = best_f

        while evals_used < budget:
            # Check time
            if time.time() > deadline - 30:
                break

            # Adapt F based on success rate (simple heuristic)
            if generation % 10 == 0 and generation > 0:
                # If best hasn't improved recently, increase F
                if stagnation_count > 5:
                    F = min(1.0, F + 0.1)
                else:
                    F = max(0.4, F - 0.05)

            for i in range(pop_size):
                if evals_used >= budget:
                    break
                if time.time() > deadline - 30:
                    break

                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)

                # DE/rand/1 mutation
                v = pop[r1] + F * (pop[r2] - pop[i])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = func(trial)
                evals_used += 1

                # Selection
                if ft <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

            generation += 1

        # Nelder-Mead polish on the best point
        if time.time() < deadline - 10:
            try:
                x0 = best_x.copy()
                # Use Nelder-Mead with bounded box
                result = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': 100,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
