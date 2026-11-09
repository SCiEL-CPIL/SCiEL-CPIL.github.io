"""Differential Evolution + CMA-ES hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution for global exploration + CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_differential_evolution(x0, max_calls, pop_size=None):
            """Differential Evolution (DE/rand/1/bin) for global search."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            if pop_size is None:
                pop_size = max(10, min(30, max_calls // 5))

            rng = np.random.default_rng(42)

            # Initialize population around x0 and random points
            population = []
            # Seed with x0
            population.append(np.array(x0, dtype=np.float64))
            # Add diverse initial points
            for _ in range(pop_size - 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Mix of local perturbations and global random
                if rng.random() < 0.3:
                    x_init = np.clip(x0 + rng.normal(0, 1.0, dim), -5, 5)
                else:
                    x_init = rng.uniform(-4.5, 4.5, dim)
                population.append(x_init)

            # Evaluate initial population
            pop_fitness = []
            for i in range(len(population)):
                if calls_made >= budget or time.time() > deadline:
                    break
                f_val = evaluate(population[i])
                pop_fitness.append(f_val)

            if len(population) < 5:
                return

            population = np.array(population[:len(pop_fitness)])
            pop_fitness = np.array(pop_fitness[:len(population)])

            # DE parameters
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate

            max_generations = max(1, max_calls // len(population))

            for gen in range(max_generations):
                if calls_made >= budget or time.time() > deadline:
                    break

                n_pop = len(population)
                if n_pop < 5:
                    break

                new_population = population.copy()
                new_fitness = pop_fitness.copy()

                for i in range(n_pop):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Select three distinct random indices different from i
                    candidates = [j for j in range(n_pop) if j != i]
                    if len(candidates) < 3:
                        break
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                    # Differential mutation
                    mutant = population[r1] + F * (population[r2] - population[r3])

                    # Binomial crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]

                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)

                    # Evaluate trial
                    f_trial = evaluate(trial)

                    # Selection
                    if f_trial <= pop_fitness[i]:
                        new_population[i] = trial
                        new_fitness[i] = f_trial

                population = new_population
                pop_fitness = new_fitness

                # Track best
                idx_best = np.argmin(pop_fitness)
                if pop_fitness[idx_best] < f_best:
                    f_best = float(pop_fitness[idx_best])
                    x_best = population[idx_best].copy()

        def run_cmaes(x0, max_calls):
            """Run CMA-ES for local refinement using scipy."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 10 or time.time() > deadline:
                return

            local_x = np.array(x0, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        def run_polish(x0, max_calls):
            """High-precision polish."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 5 or time.time() > deadline:
                return
            run_cmaes(x0, max_calls)

        # Phase 1: Differential Evolution for global exploration
        de_calls = int(budget * 0.65)
        if time.time() < deadline:
            run_differential_evolution(np.zeros(dim), de_calls, pop_size=20)

        # Phase 2: CMA-ES refinement on best point
        remaining = budget - calls_made
        if remaining >= 15 and x_best is not None and time.time() < deadline:
            cma_calls = min(remaining, 80)
            run_cmaes(x_best.copy(), cma_calls)

        # Phase 3: Second DE run from best point with smaller population for fine search
        remaining = budget - calls_made
        if remaining >= 20 and x_best is not None and time.time() < deadline:
            de_calls2 = min(remaining, 50)
            run_differential_evolution(x_best.copy(), de_calls2, pop_size=10)

        # Phase 4: Final polish
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            polish_calls = min(remaining, 30)
            run_polish(x_best.copy(), polish_calls)

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
