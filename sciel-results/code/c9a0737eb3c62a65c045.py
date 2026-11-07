"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based global optimizer with adaptive parameters, restarts, and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def call_func(x):
            nonlocal total_calls
            total_calls += 1
            return float(func(x))

        # LHS initialization for diversity
        def lhs_sample(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[perm, d] = np.linspace(0, 1, n, endpoint=False) + rng.uniform(0, 1/n, n)
            return points * 10.0 - 5.0  # map to [-5, 5]

        # DE/rand/1 with adaptive mutation factor
        def de_step(pop, pop_f, best_idx, F, CR, rng):
            n = len(pop)
            new_pop = np.copy(pop)
            new_f = np.copy(pop_f)
            for i in range(n):
                if total_calls >= budget:
                    break
                # Select 3 distinct indices different from i
                r1, r2, r3 = i, i, i
                while r1 == i:
                    r1 = rng.integers(0, n)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, n)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, n)

                v = pop[r1] + F * (pop[r2] - pop[r3])
                # Binary crossover
                j_rand = rng.integers(0, dim)
                trial = np.copy(pop[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                ft = call_func(trial)
                if ft < pop_f[i]:
                    new_pop[i] = trial
                    new_f[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        best_idx = i
                # Update F adaptively (simple: if improved, keep F; else reduce slightly)
            return new_pop, new_f, best_idx

        # Nelder-Mead polish
        def nelder_mead_polish(x0, f0, max_iter=50):
            nonlocal total_calls, best_x, best_f
            dim = self.dim
            simplex = [x0.copy()]
            f_vals = [f0]
            # Build simplex
            for i in range(dim):
                pt = x0.copy()
                step = 0.1 if abs(x0[i]) > 1e-8 else 0.1
                pt[i] += step
                pt = np.clip(pt, -5, 5)
                if np.allclose(pt, x0):
                    pt[i] -= 2 * step
                    pt = np.clip(pt, -5, 5)
                simplex.append(pt)
                f_vals.append(call_func(pt))

            alpha, gamma, rho = 1.0, 2.0, 0.5

            for _ in range(max_iter):
                if total_calls >= budget or time.time() > deadline:
                    break
                # Sort by function value
                idx = np.argsort(f_vals)
                simplex = [simplex[i].copy() for i in idx]
                f_vals = [f_vals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call_func(x_r)

                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call_func(x_c)
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_vals[i] = call_func(simplex[i])

                # Track best
                min_idx = np.argmin(f_vals)
                if f_vals[min_idx] < best_f:
                    best_f = f_vals[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Main loop: DE with restarts
        pop_size = 20
        F = 0.8
        CR = 0.5

        # Initial population using LHS
        pop = lhs_sample(pop_size, dim)
        pop_f = np.array([call_func(p) for p in pop])

        # Track best
        init_best_idx = np.argmin(pop_f)
        best_f = pop_f[init_best_idx]
        best_x = pop[init_best_idx].copy()

        stagnation_counter = 0
        no_improve = 0
        prev_best = best_f

        while total_calls < budget and time.time() < deadline:
            # DE step
            pop, pop_f, best_idx = de_step(pop, pop_f, np.argmin(pop_f), F, CR, rng)

            # Update best
            cur_best_idx = np.argmin(pop_f)
            cur_best_f = pop_f[cur_best_idx]
            if cur_best_f < best_f:
                best_f = cur_best_f
                best_x = pop[cur_best_idx].copy()
                no_improve = 0
                stagnation_counter = 0
            else:
                no_improve += 1
                stagnation_counter += 1

            # Adaptive F and CR
            if no_improve > 10:
                F = max(0.3, F * 0.9)
                CR = min(0.9, CR + 0.02)
            elif no_improve > 30:
                F = max(0.2, F * 0.85)

            # Restart if stagnated
            if stagnation_counter > 50:
                # Perturb population around best
                for i in range(pop_size):
                    if i == cur_best_idx:
                        continue
                    pop[i] = best_x + rng.normal(0, 0.5, dim)
                    pop[i] = np.clip(pop[i], -5, 5)
                    pop_f[i] = call_func(pop[i])
                    if pop_f[i] < best_f:
                        best_f = pop_f[i]
                        best_x = pop[i].copy()
                        cur_best_idx = i
                stagnation_counter = 0
                no_improve = 0
                F = 0.8
                CR = 0.5

            # Check for early termination if converged
            if best_f < -99.9 and no_improve > 100:
                break

        # Final polish with Nelder-Mead if budget allows
        if total_calls < budget - 10 and time.time() < deadline - 5:
            best_x, best_f = nelder_mead_polish(best_x, best_f, max_iter=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
