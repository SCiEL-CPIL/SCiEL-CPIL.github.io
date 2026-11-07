"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population-based DE
        pop_size = 20
        CR = 0.8  # crossover rate
        F = 0.8   # differential weight

        # Initialize population
        population = np.random.uniform(-4, 4, size=(pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls_made = pop_size

        # Track global best
        global_best_idx = np.argmin(fitness)
        global_best_x = population[global_best_idx].copy()
        global_best_fx = fitness[global_best_idx]

        # Adaptive parameters
        no_improve_count = 0
        max_no_improve = 40
        current_F = F
        current_CR = CR

        # Use 80% of budget for DE, rest for local refinement
        de_budget = int(budget * 0.8)

        while calls_made < de_budget:
            if time.time() > deadline:
                break

            # Check for stagnation
            if no_improve_count >= max_no_improve:
                # Adaptive restart: diversify population
                # Keep the best few, replace rest with random
                keep = 3
                if global_best_x is not None:
                    # Ensure global best is in population
                    min_idx = np.argmin(fitness)
                    if fitness[min_idx] > global_best_fx:
                        population[min_idx] = global_best_x
                        fitness[min_idx] = global_best_fx

                for i in range(keep, pop_size):
                    if calls_made >= de_budget:
                        break
                    if rng.random() < 0.5:
                        # Local perturbation around best
                        population[i] = global_best_x + 2.0 * rng.standard_normal(dim)
                    else:
                        # Global random
                        population[i] = np.random.uniform(-4, 4, size=dim)
                    population[i] = np.clip(population[i], -5, 5)
                    fitness[i] = func(population[i])
                    calls_made += 1
                    if fitness[i] < global_best_fx:
                        global_best_fx = fitness[i]
                        global_best_x = population[i].copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                if calls_made >= de_budget:
                    break

                # Reset adaptive parameters
                no_improve_count = 0
                current_F = F
                current_CR = CR
                max_no_improve = 40

            # DE/rand/1/bin iteration
            for i in range(pop_size):
                if calls_made >= de_budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # Create mutant vector
                v = population[r1] + current_F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Binary crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < current_CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                trial_fx = func(trial)
                calls_made += 1

                # Selection
                if trial_fx <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fx

                    # Update global best
                    if trial_fx < global_best_fx:
                        global_best_fx = trial_fx
                        global_best_x = trial.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

                # Adapt F and CR based on progress
                if no_improve_count % 15 == 0 and no_improve_count > 0:
                    # Slightly adjust parameters to escape local optima
                    current_F = np.clip(current_F + 0.1 * rng.standard_normal(), 0.5, 1.2)
                    current_CR = np.clip(current_CR + 0.05 * rng.standard_normal(), 0.5, 1.0)

        # Final local refinement using Scipy Nelder-Mead
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = budget - calls_made

            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return 1e10
                calls_made += 1
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining_budget, 'maxfev': remaining_budget,
                                           'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
