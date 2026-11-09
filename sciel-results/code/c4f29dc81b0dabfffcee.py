"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE with archive-guided search, adaptive mutation, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 320.0

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # --- Initialization: corners + structured + random ---
        pop_size = min(40, max(15, budget // 5))

        corners = []
        for i in range(2 ** dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(corner)

        structured_extra = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
            lo + 0.1 * span,
            lo + 0.9 * span,
        ]

        init_points = []
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_idx = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_idx])

        for sp in structured_extra:
            if len(init_points) < pop_size:
                init_points.append(sp)

        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Archive management ---
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            new_ax = []
            new_af = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_ax.append(ax)
                    new_af.append(af)
            new_ax.append(x.copy())
            new_af.append(f.copy())
            archive_X = new_ax
            archive_F = new_af

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Main DE loop with adaptive mutation ---
        CR = 0.7
        F_mut = 0.5
        best_sum = np.min(np.sum(F, axis=1))
        stagnation_count = 0
        stagnation_window = 20

        polish_budget = max(25, budget // 8)
        main_budget = budget - polish_budget

        while used < main_budget:
            if time.time() - start_time > DEADLINE:
                break

            # Adaptive mutation scale based on stagnation
            if stagnation_count > stagnation_window * 2:
                F_mut_current = 0.8
            elif stagnation_count > stagnation_window:
                F_mut_current = 0.6
            else:
                F_mut_current = 0.5

            # Select base: prefer archive points for exploitation
            if len(archive_X) >= 3 and rng.random() < 0.6:
                base_idx = rng.integers(len(archive_X))
                base_x = archive_X[base_idx]
                base_in_pop = -1
            else:
                base_idx = rng.integers(pop_size)
                base_x = X[base_idx]
                base_in_pop = base_idx

            # Select two other parents from archive or population
            if len(archive_X) >= 3:
                other_idx = rng.choice(len(archive_X), size=2, replace=False)
                x1 = archive_X[other_idx[0]]
                x2 = archive_X[other_idx[1]]
            else:
                other_idx = rng.choice(pop_size, size=2, replace=False)
                x1 = X[other_idx[0]]
                x2 = X[other_idx[1]]

            # Mutation
            mutant = base_x + F_mut_current * (x1 - x2)
            mutant = np.clip(mutant, lo, hi)

            # Binomial crossover
            trial = base_x.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Track stagnation
            if np.sum(f_trial) < best_sum:
                best_sum = np.sum(f_trial)
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update archive
            update_archive(trial, f_trial)

            # Population replacement
            if base_in_pop >= 0:
                base_f = F[base_in_pop]
                if np.all(f_trial <= base_f) and np.any(f_trial < base_f):
                    X[base_in_pop] = trial
                    F[base_in_pop] = f_trial
                elif not (np.all(base_f <= f_trial) and np.any(base_f < f_trial)):
                    if np.sum(f_trial) < np.sum(base_f) or rng.random() < 0.4:
                        X[base_in_pop] = trial
                        F[base_in_pop] = f_trial

            # Periodic diversity injection
            if used % 15 == 0 and used < main_budget:
                if time.time() - start_time > DEADLINE:
                    break
                rand_x = lo + span * rng.random(dim)
                rand_f = np.array(problem(rand_x))
                used += 1
                update_archive(rand_x, rand_f)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # --- Polish phase: local search around top archive points ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

            sums = np.array([np.sum(af) for af in archive_F])
            top_idx = np.argsort(sums)[:min(12, len(archive_X))]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Multi-scale perturbations
                scales = [0.05, 0.1, 0.15]
                for scale in scales:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break

                    for _ in range(min(3, budget - used)):
                        if time.time() - start_time > DEADLINE:
                            break
                        sigma = scale * span
                        perturbed = center + rng.normal(0, sigma, dim)
                        perturbed = np.clip(perturbed, lo, hi)

                        f_pert = np.array(problem(perturbed))
                        used += 1

                        if np.any(f_pert < center_f) and not np.all(center_f <= f_pert):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert

        return X
# EVOLVE-BLOCK-END
