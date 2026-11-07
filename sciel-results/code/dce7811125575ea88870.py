"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: random init + DE mutation + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial random sampling (30% of budget)
        n_init = max(10, int(budget * 0.30))
        n_init = min(n_init, budget)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        def is_dominated(f, F_arr):
            """Check if f is dominated by any point in F_arr."""
            # f is dominated if there exists g in F_arr such that g <= f and g != f
            if np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1)):
                return True
            return False

        def pareto_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        # Phase 2: Differential Evolution on remaining budget
        n_de = budget - n_init
        if n_de > 0:
            # Population size for DE: keep it manageable
            pop_size = min(20, max(10, n_de // 3))
            # If we have enough evaluations, use a larger population
            if n_de >= 100:
                pop_size = min(25, max(15, n_de // 4))

            # Initialize DE population from current best points
            # Keep the non-dominated points as initial DE population
            pm = pareto_mask(F)
            nd_points = X[pm]
            nd_F = F[pm]

            # If not enough non-dominated points, add random ones
            if len(nd_points) < pop_size:
                n_add = pop_size - len(nd_points)
                X_add = lo + (hi - lo) * rng.random((n_add, dim))
                F_add = np.array([problem(x) for x in X_add])
                used += n_add
                nd_points = np.vstack([nd_points, X_add])
                nd_F = np.vstack([nd_F, F_add])
            else:
                # Shuffle and take first pop_size
                idx = rng.permutation(len(nd_points))[:pop_size]
                nd_points = nd_points[idx]
                nd_F = nd_F[idx]

            de_X = nd_points.copy()
            de_F = nd_F.copy()
            F_cr = 0.8  # Crossover rate
            F_mut = 0.5  # Mutation scale

            # DE loop
            while used < budget:
                if time.time() > deadline - 30:
                    break

                # Check time and break if close to deadline
                if time.time() > deadline - 5:
                    break

                # For each individual in population, generate a mutant
                gen_X = []
                gen_F = []

                for i in range(len(de_X)):
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Select three distinct random individuals different from i
                    candidates = [j for j in range(len(de_X)) if j != i]
                    if len(candidates) < 3:
                        continue
                    a, b, c = rng.choice(candidates, size=3, replace=False)

                    # DE/rand/1 mutation
                    mutant = de_X[a] + F_mut * (de_X[b] - de_X[c])
                    mutant = np.clip(mutant, lo, hi)

                    # Binomial crossover
                    cross_mask = rng.random(dim) < F_cr
                    if not np.any(cross_mask):
                        cross_mask[rng.integers(dim)] = True
                    trial = np.where(cross_mask, mutant, de_X[i])
                    trial = np.clip(trial, lo, hi)

                    f_trial = problem(trial)
                    used += 1

                    # Selection: keep the better one
                    if is_dominated(de_F[i], np.array([f_trial])):
                        # de_F[i] is dominated by f_trial, replace
                        gen_X.append(trial)
                        gen_F.append(f_trial)
                    elif is_dominated(f_trial, np.array([de_F[i]])):
                        # trial is dominated, keep original
                        gen_X.append(de_X[i])
                        gen_F.append(de_F[i])
                    else:
                        # Neither dominates; keep the one with better sum (or random)
                        if np.sum(f_trial) < np.sum(de_F[i]):
                            gen_X.append(trial)
                            gen_F.append(f_trial)
                        else:
                            gen_X.append(de_X[i])
                            gen_F.append(de_F[i])

                if gen_X:
                    de_X = np.array(gen_X)
                    de_F = np.array(gen_F)

        # Phase 3: Local polish on top Pareto points
        # Combine all evaluated points
        all_X = []
        all_F = []

        # Add initial points
        all_X.append(X)
        all_F.append(F)

        # Add DE population points
        if n_de > 0 and len(de_X) > 0:
            all_X.append(de_X)
            all_F.append(de_F)

        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)

        # Remove dominated points from the archive
        pm = pareto_mask(all_F)
        archive_X = all_X[pm]
        archive_F = all_F[pm]

        # Polish top 5 Pareto points using Nelder-Mead on each objective separately
        # Actually, for multi-objective, we do a local search that tries to improve
        # hypervolume contribution. We'll use a simple coordinate descent / Nelder-Mead
        # on a scalarized objective for each of the top points.

        n_polish = min(5, len(archive_X))
        if n_polish > 0 and time.time() < deadline - 10:
            # Sort by sum of objectives (lower is better) to pick top candidates
            sums = np.sum(archive_F, axis=1)
            top_idx = np.argsort(sums)[:n_polish]

            for idx in top_idx:
                if time.time() > deadline - 10:
                    break
                if used >= budget:
                    break

                x0 = archive_X[idx].copy()
                f0 = archive_F[idx].copy()

                # Try to improve each objective independently via local search
                # Use a small step size and coordinate-wise exploration
                step_size = 0.02 * (hi - lo)
                improved = True
                max_iters = 50

                for obj_idx in range(n_obj):
                    if time.time() > deadline - 10:
                        break
                    if used >= budget:
                        break

                    # Try to minimize objective obj_idx while keeping others reasonable
                    # Use a simple line search along each coordinate
                    for coord in range(dim):
                        if used >= budget or time.time() > deadline - 10:
                            break

                        # Try positive and negative steps
                        for direction in [-1, 1]:
                            if used >= budget or time.time() > deadline - 10:
                                break

                            x_try = x0.copy()
                            x_try[coord] += direction * step_size
                            x_try = np.clip(x_try, lo, hi)

                            f_try = problem(x_try)
                            used += 1

                            # Accept if it improves the target objective without
                            # significantly worsening others
                            if f_try[obj_idx] < f0[obj_idx]:
                                # Check if other objectives haven't worsened too much
                                other_worse = np.any(f_try[~np.array([obj_idx])] > f0[~np.array([obj_idx])] * 1.05)
                                if not other_worse:
                                    x0 = x_try
                                    f0 = f_try
                                    improved = True

                        if improved:
                            # Reduce step size for next iteration
                            step_size *= 0.7
                            improved = False

                # Add polished point to archive if it's non-dominated
                if not is_dominated(f0, archive_F):
                    archive_X = np.vstack([archive_X, x0])
                    archive_F = np.vstack([archive_F, f0])

        # Final cleanup: remove dominated points from final archive
        if len(archive_F) > 1:
            pm_final = pareto_mask(archive_F)
            archive_X = archive_X[pm_final]
            archive_F = archive_F[pm_final]

        return archive_X
# EVOLVE-BLOCK-END
