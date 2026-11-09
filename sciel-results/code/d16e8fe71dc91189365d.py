"""Optimized solver for BBOB f9 (rotated Rosenbrock) using DE + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE global search + L-BFGS-B polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 5.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution for global search
        # DE is population-based and robust to unknown rotation/shift
        de_population_size = 20
        max_de_generations = 20
        de_call_budget = min(budget - 50, 400)  # Reserve 50 for polish

        # Initialize DE population
        population = [rng.uniform(-4.5, 4.5, dim) for _ in range(de_population_size)]
        population_f = [counted_func(p) for p in population]

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        for gen in range(max_de_generations):
            if not has_budget() or call_count[0] >= de_call_budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(population_f)
            sorted_pop = [population[i] for i in sorted_idx]
            sorted_f = [population_f[i] for i in sorted_idx]

            # Keep best individual
            best_individual = sorted_pop[0].copy()
            best_fitness = sorted_f[0]

            # Generate new population
            new_population = []
            new_population_f = []

            for i in range(de_population_size):
                if not has_budget() or call_count[0] >= de_call_budget:
                    break

                # DE/rand/1 mutation
                # Select three distinct individuals different from i
                a, b, c = rng.choice(de_population_size, size=3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = rng.choice(de_population_size, size=3, replace=False)

                v = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
                v = np.clip(v, -5.0, 5.0)

                # Crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = sorted_pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5.0, 5.0)
                trial_f = counted_func(trial)

                # Selection
                if trial_f < sorted_f[i]:
                    new_population.append(trial)
                    new_population_f.append(trial_f)
                else:
                    new_population.append(sorted_pop[i])
                    new_population_f.append(sorted_f[i])

            # Update population if we have enough
            if len(new_population) == de_population_size:
                population = new_population
                population_f = new_population_f

            # Update global best
            if best_fitness < best_f[0]:
                best_f[0] = best_fitness
                best_x[0] = best_individual.copy()

        # Use tracked best if better
        if best_x[0] is not None and best_f[0] < np.inf:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0].copy()
        else:
            overall_best_f = np.inf
            overall_best_x = np.zeros(dim)

        # Phase 2: High-precision L-BFGS-B polish on best candidate
        if has_budget() and remaining_time() > 20.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Final ultra-tight BFGS polish
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-16, 'disp': False}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B with even tighter tolerances
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
