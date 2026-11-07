"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Population-based DE with adaptive parameters, multi-start Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Differential Evolution (DE/rand/1/bin)
        # Use ~60% of budget for global exploration
        de_budget = int(budget * 0.6)
        pop_size = min(30, max(12, de_budget // 8))

        # Initialize population: mix of global random and local perturbations
        # First, evaluate a few random points to get a rough sense of the landscape
        initial_points = []
        for i in range(min(5, pop_size)):
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            calls_made += 1
            initial_points.append((x, fx))
            try_update(x, fx)
            if check_deadline():
                break

        # Initialize full population
        population = np.zeros((pop_size, dim))
        fitness = np.zeros(pop_size)

        # Seed with initial points
        for i in range(min(len(initial_points), pop_size)):
            population[i] = initial_points[i][0]
            fitness[i] = initial_points[i][1]

        # Fill remaining slots
        for i in range(len(initial_points), pop_size):
            if calls_made >= de_budget or check_deadline():
                break
            if rng.random() < 0.4:
                # Local perturbation around current best
                if global_best_x is not None:
                    population[i] = global_best_x + 1.5 * rng.standard_normal(dim)
                else:
                    population[i] = rng.uniform(-4, 4, size=dim)
            else:
                # Global random
                population[i] = rng.uniform(-4, 4, size=dim)
            population[i] = np.clip(population[i], -5, 5)
            fitness[i] = func(population[i])
            calls_made += 1
            try_update(population[i], fitness[i])

        # DE parameters
        F = 0.8
        CR = 0.7
        no_improve_count = 0
        max_no_improve = 40
        F_history = []
        CR_history = []
        F_history.append(F)
        CR_history.append(CR)

        # Main DE loop
        while calls_made < de_budget and not check_deadline():
            # Adaptive parameter update every few generations
            if len(F_history) > 5 and len(F_history) % 5 == 0:
                # Use best parameters from history
                best_idx = np.argmin(fitness[:len(F_history)]) if len(F_history) <= len(fitness) else 0
                # Simple adaptation: nudge toward values that worked
                if no_improve_count < max_no_improve // 2:
                    F = np.clip(F + 0.05 * rng.standard_normal(), 0.5, 1.2)
                    CR = np.clip(CR + 0.03 * rng.standard_normal(), 0.5, 1.0)
                else:
                    F = 0.8
                    CR = 0.7
                F_history.append(F)
                CR_history.append(CR)

            # DE/rand/1/bin iteration
            for i in range(pop_size):
                if calls_made >= de_budget or check_deadline():
                    break

                # Select three distinct random individuals
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # Create mutant vector
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Binary crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                trial_fx = func(trial)
                calls_made += 1

                # Selection
                if trial_fx <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fx
                    if try_update(trial, trial_fx):
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            # Stagnation handling
            if no_improve_count >= max_no_improve:
                # Diversify: replace worst half with new random points
                half = pop_size // 2
                order = np.argsort(fitness)
                for i in range(half, pop_size):
                    if calls_made >= de_budget or check_deadline():
                        break
                    if rng.random() < 0.5:
                        if global_best_x is not None:
                            population[i] = global_best_x + 2.5 * rng.standard_normal(dim)
                        else:
                            population[i] = rng.uniform(-4, 4, size=dim)
                    else:
                        population[i] = rng.uniform(-4, 4, size=dim)
                    population[i] = np.clip(population[i], -5, 5)
                    fitness[i] = func(population[i])
                    calls_made += 1
                    try_update(population[i], fitness[i])

                no_improve_count = 0
                max_no_improve = 30

        # Phase 2: Multi-start Nelder-Mead local refinement
        # Use remaining budget for local refinement from multiple starting points
        remaining_budget = budget - calls_made
        if global_best_x is not None and remaining_budget > 10 and not check_deadline():
            # Get top K diverse points from population
            # Sort by fitness
            order = np.argsort(fitness)
            
            # Select diverse starting points: best + some random from top half
            start_points = []
            start_points.append((global_best_x.copy(), global_best_fx))
            
            # Add some diverse points from the population
            for idx in order[1:]:
                if len(start_points) >= min(5, pop_size // 3):
                    break
                # Check diversity: ensure it's not too close to existing start points
                x_candidate = population[idx].copy()
                is_diverse = True
                for sp in start_points:
                    if np.linalg.norm(x_candidate - sp[0]) < 1.0:
                        is_diverse = False
                        break
                if is_diverse:
                    start_points.append((x_candidate, fitness[idx]))
            
            # Add a few random points for diversity
            for _ in range(min(3, remaining_budget // 20)):
                if calls_made >= budget or check_deadline():
                    break
                x_rand = rng.uniform(-4, 4, size=dim)
                fx_rand = func(x_rand)
                calls_made += 1
                try_update(x_rand, fx_rand)
                start_points.append((x_rand, fx_rand))
            
            # Sort start points by fitness
            start_points.sort(key=lambda x: x[1])
            
            # Allocate budget to each start point
            num_starts = len(start_points)
            budget_per_start = max(5, remaining_budget // num_starts)
            
            for sp_x, sp_fx in start_points:
                if calls_made >= budget or check_deadline():
                    break
                
                local_budget = min(budget_per_start, budget - calls_made)
                if local_budget < 5:
                    continue
                
                def wrapped_func(x, _budget_counter=[0]):
                    nonlocal calls_made
                    if calls_made >= budget or check_deadline():
                        return 1e10
                    calls_made += 1
                    return func(np.clip(x, -5, 5))
                
                try:
                    result = minimize(wrapped_func, sp_x, method='Nelder-Mead',
                                      options={'maxiter': local_budget, 'maxfev': local_budget,
                                               'xatol': 1e-10, 'fatol': 1e-10,
                                               'adaptive': True})
                    if result.fun < global_best_fx:
                        global_best_fx = result.fun
                        global_best_x = result.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Final fine-tuning around global best if budget remains
        if global_best_x is not None and calls_made < budget and not check_deadline():
            remaining = budget - calls_made
            if remaining >= 10:
                # Do a small local search with very tight tolerances
                def final_func(x):
                    nonlocal calls_made
                    if calls_made >= budget or check_deadline():
                        return 1e10
                    calls_made += 1
                    return func(np.clip(x, -5, 5))
                
                try:
                    result = minimize(final_func, global_best_x, method='Nelder-Mead',
                                      options={'maxiter': remaining, 'maxfev': remaining,
                                               'xatol': 1e-12, 'fatol': 1e-12,
                                               'adaptive': True})
                    if result.fun < global_best_fx:
                        global_best_fx = result.fun
                        global_best_x = result.x.copy()
                except Exception:
                    pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
