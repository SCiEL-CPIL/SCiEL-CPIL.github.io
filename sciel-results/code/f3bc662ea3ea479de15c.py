"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE/rand/1/bin with adaptive step-size and coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Track best solution
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        calls = 1

        def check_time():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls
            if calls >= budget:
                return None
            val = func(x)
            calls += 1
            return val

        # Phase 1: Differential Evolution
        # Population size
        N_p = min(20, budget - 1)
        if N_p < 4:
            N_p = max(2, budget - 1)

        F = 0.8
        CR = 0.5

        # Initialize population
        pop = np.empty((N_p, dim))
        pop[0] = best_x
        for i in range(1, N_p):
            if not check_time():
                break
            pop[i] = rng.uniform(-5, 5, dim)
            f_val = call_func(pop[i])
            if f_val is None:
                break
            if f_val < best_f:
                best_f = f_val
                best_x = pop[i].copy()

        # Track fitness for population
        fit = np.empty(N_p)
        fit[0] = best_f
        for i in range(1, N_p):
            if calls > i:
                # We already evaluated these
                pass
            # Re-evaluate if needed (simpler: just keep track)
            # Actually let's restructure: store fitness properly

        # Let's redo population init more carefully
        pop = np.empty((N_p, dim))
        fit = np.empty(N_p)
        pop[0] = best_x
        fit[0] = best_f
        for i in range(1, N_p):
            if not check_time() or calls >= budget:
                N_p = i
                break
            pop[i] = rng.uniform(-5, 5, dim)
            f_val = call_func(pop[i])
            if f_val is None:
                N_p = i
                break
            fit[i] = f_val
            if f_val < best_f:
                best_f = f_val
                best_x = pop[i].copy()

        # DE loop
        generation = 0
        stagnation_count = 0
        last_best_f = best_f

        while calls < budget and check_time():
            generation += 1
            improved_this_gen = False

            for i in range(N_p):
                if calls >= budget or not check_time():
                    break

                # Select three distinct random individuals
                while True:
                    a, b, c = rng.integers(0, N_p, 3)
                    if len(set([a, b, c, i])) == 4:
                        break

                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover: binomial
                r = rng.uniform()
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if r <= CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate
                f_val = call_func(trial)
                if f_val is None:
                    break

                # Selection
                if f_val <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_val
                    improved_this_gen = True
                    if f_val < best_f:
                        best_f = f_val
                        best_x = trial.copy()

            # Check for stagnation
            if best_f < last_best_f - 1e-10:
                stagnation_count = 0
                last_best_f = best_f
            else:
                stagnation_count += 1

            # Adaptive F
            if stagnation_count > 10:
                F = max(0.5, F - 0.05)
                stagnation_count = 0

            # Occasionally re-inject diversity
            if stagnation_count > 20 and N_p > 4:
                idx = rng.integers(0, N_p)
                if calls < budget and check_time():
                    pop[idx] = best_x + rng.standard_normal(dim) * 0.5
                    pop[idx] = np.clip(pop[idx], -5, 5)
                    f_val = call_func(pop[idx])
                    if f_val is not None:
                        fit[idx] = f_val
                        if f_val < best_f:
                            best_f = f_val
                            best_x = pop[idx].copy()
                stagnation_count = 0

        # Phase 2: Coordinate Descent Polish
        # Use remaining budget for local refinement
        if calls < budget and check_time():
            # Try coordinate descent with line search
            for coord in range(dim):
                if calls >= budget or not check_time():
                    break
                # Binary search along each coordinate
                for _ in range(3):
                    step = 0.1
                    for _ in range(20):
                        if calls >= budget or not check_time():
                            break
                        # Try moving in positive and negative directions
                        x_plus = best_x.copy()
                        x_plus[coord] = np.clip(best_x[coord] + step, -5, 5)
                        f_plus = call_func(x_plus)
                        if f_plus is None:
                            break

                        x_minus = best_x.copy()
                        x_minus[coord] = np.clip(best_x[coord] - step, -5, 5)
                        f_minus = call_func(x_minus)
                        if f_minus is None:
                            break

                        if f_plus < best_f:
                            best_x = x_plus
                            best_f = f_plus
                        elif f_minus < best_f:
                            best_x = x_minus
                            best_f = f_minus
                        else:
                            step *= 0.5
                            break
                        step *= 0.5

        # Final check
        if best_f is None:
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
