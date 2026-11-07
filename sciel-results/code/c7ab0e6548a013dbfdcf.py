"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid DE + Multi-start Nelder-Mead local search for BBOB f4 (Büche-Rastrigin).
    Uses a time-based deadline guard to stay within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # --- Phase 1: Differential Evolution (DE) for global exploration ---
        # DE population size
        pop_size = max(10, 4 * n)
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Initialize population uniformly in [-5, 5]^n
        pop = np.random.uniform(-5, 5, (pop_size, n))
        # Evaluate initial population
        fit = np.array([func(x) for x in pop])

        # Track best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        calls = pop_size

        # DE main loop
        # Each generation uses pop_size evaluations
        # We'll run DE until we've used about 60% of budget or hit deadline
        de_budget = int(budget * 0.6)
        gen = 0
        while calls < de_budget and time.time() < deadline:
            gen += 1
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if len(set([i, r1, r2, r3])) == 4:
                        break

                # Create trial vector
                v = pop[i] + F * (pop[r1] - pop[r2])
                # Crossover
                j_rand = np.random.randint(n)
                trial = pop[i].copy()
                for j in range(n):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls += 1

                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        # --- Phase 2: Multi-start Nelder-Mead local refinement ---
        # Use remaining budget for local refinement
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            # Try Nelder-Mead from best point and a few other good points
            # Sort population by fitness
            sorted_idx = np.argsort(fit)
            # Take top few candidates for local refinement
            num_starts = min(5, pop_size)
            starts = [pop[sorted_idx[k]].copy() for k in range(num_starts)]
            starts.append(best_x.copy())

            # Nelder-Mead implementation
            def nelder_mead(func, x0, n, max_iter, deadline):
                """Simple Nelder-Mead local search."""
                # Initial simplex
                eps = 0.5
                simplex = [x0.copy()]
                for i in range(n):
                    xi = x0.copy()
                    if abs(xi[i]) > eps:
                        xi[i] += 0.05 * xi[i]
                    else:
                        xi[i] += eps
                    simplex.append(xi)

                fvals = [func(x) for x in simplex]
                calls_nm = n + 1

                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    # Sort simplex by fitness
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    # Reflection
                    centroid = np.mean(simplex[:-1], axis=0)
                    x_ref = centroid + 1.0 * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = func(x_ref)
                    calls_nm += 1

                    if f_ref < fvals[0]:
                        # Expansion
                        x_exp = centroid + 2.0 * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        calls_nm += 1
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    elif f_ref < fvals[-2]:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + 0.5 * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_nm += 1
                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                calls_nm += 1
                                if time.time() > deadline:
                                    break
                            if time.time() > deadline:
                                break

                    # Return best
                    best_i = np.argmin(fvals)
                    return simplex[best_i], fvals[best_i], calls_nm

            # Run Nelder-Mead on each start
            for start in starts:
                if calls >= budget or time.time() > deadline:
                    break
                # Estimate max iterations for NM based on remaining budget
                # Each NM iteration uses ~2-3 evaluations
                max_iter = min(50, max(5, (budget - calls) // (num_starts * 3)))
                x_nm, f_nm, calls_used = nelder_mead(func, start, n, max_iter, deadline)
                calls += calls_used
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
