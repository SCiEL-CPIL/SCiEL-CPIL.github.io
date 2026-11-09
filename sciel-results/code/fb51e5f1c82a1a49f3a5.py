"""Multi-objective optimizer for RE34 using LHS initialization and DE search with Pareto archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Iterative differential-evolution search with Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(0)

        # ---------- Initialization: Latin Hypercube Sampling for uniform coverage ----------
        pop_size = 30
        n_lh = pop_size

        # Generate Latin Hypercube samples
        lh_samples = np.zeros((n_lh, dim))
        for d in range(dim):
            strata = np.linspace(0, 1, n_lh + 1)
            for i in range(n_lh):
                lh_samples[i, d] = strata[i] + (strata[i + 1] - strata[i]) * rng.random()
        
        # Map to actual bounds
        X_init = lo + (hi - lo) * lh_samples

        # Add corners for boundary exploration (replacing some LHS points)
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )
        # Replace 8 LHS points with 8 well-spread corners for better boundary coverage
        corner_indices = [0, 1, 2, 4, 8, 16, 5, 10]
        for i, ci in enumerate(corner_indices):
            X_init[i] = corners[ci]

        # Add midpoint
        mid = (lo + hi) / 2.0
        X_init[-1] = mid

        # Evaluate initial population
        F = np.array([problem(x) for x in X_init])
        used = pop_size

        X = X_init.copy()

        # ---------- Pareto archive ----------
        archive_X = []
        archive_F = []

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, arch_F):
            for af in arch_F:
                if dominates(af, f):
                    return True
            return False

        def crowding_distance(arch_F):
            """Compute crowding distance for each member of the archive."""
            n = len(arch_F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                obj = np.array([af[m] for af in arch_F])
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = obj[order[-1]] - obj[order[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / span
            return cd

        def update_archive(x, f, arch_X, arch_F):
            """Add x/f to archive if non-dominated, removing dominated members."""
            if is_dominated(f, arch_F):
                return arch_X, arch_F
            for j, af in enumerate(arch_F):
                if np.allclose(f, af, atol=1e-12):
                    return arch_X, arch_F
            new_X = []
            new_F = []
            for ax, af in zip(arch_X, arch_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            return new_X, new_F

        # Add initial population to archive
        for i in range(len(X)):
            archive_X, archive_F = update_archive(X[i], F[i], archive_X, archive_F)

        # ---------- Main search loop ----------
        # Adaptive DE parameters: start with exploration, shift to exploitation
        base_F = 0.7
        base_CR = 0.9
        n_iter = 0
        total_iter = budget - pop_size
        stagnation_counter = 0
        best_archive_size = len(archive_F)

        while used < budget:
            if time.time() - start_time > deadline:
                break

            n_iter += 1
            progress = n_iter / max(1, total_iter)
            
            # Adaptive parameters: decrease F, increase CR as we progress
            F_factor = base_F * (1.0 - 0.3 * progress)  # 0.7 -> 0.49
            CR = base_CR + 0.1 * progress  # 0.9 -> 1.0

            # Select parent from archive using crowding distance (weighted toward crowded regions)
            if len(archive_X) > 1:
                cd = crowding_distance(archive_F)
                # Use tournament selection based on crowding distance
                i1 = rng.integers(len(archive_X))
                i2 = rng.integers(len(archive_X))
                parent_idx = i1 if cd[i1] >= cd[i2] else i2
                parent = archive_X[parent_idx]
            elif len(archive_X) == 1:
                parent = archive_X[0]
            else:
                parent = X[rng.integers(pop_size)]

            # DE mutation: select 3 distinct random indices from population
            idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)

            # Create trial vector
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            # Binary crossover with parent
            mask = rng.random(dim) < CR
            trial = np.where(mask, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Check if this improves the archive
            prev_archive_size = len(archive_F)
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)
            
            if len(archive_F) > best_archive_size:
                best_archive_size = len(archive_F)
                stagnation_counter = 0
            elif len(archive_F) == prev_archive_size:
                stagnation_counter += 1
            else:
                stagnation_counter = max(0, stagnation_counter - 1)

            # Update population with diversity-aware replacement
            # Instead of always replacing worst, use a strategy that maintains diversity
            if rng.random() < 0.7:
                # Replace worst member if trial is better
                worst_idx = np.argmax(np.sum(F, axis=1))
                if not dominates(F[worst_idx], ft):
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                elif rng.random() < 0.15:
                    # Occasionally replace even if not better, for diversity
                    X[worst_idx] = trial
                    F[worst_idx] = ft
            else:
                # Replace a random member to maintain diversity
                rand_idx = rng.integers(pop_size)
                X[rand_idx] = trial
                F[rand_idx] = ft

            # Periodically refresh population with new stratified points to maintain diversity
            if used % 40 == 0 and used < budget - 15:
                idx = rng.integers(pop_size)
                # Use stratified sampling for diversity
                new_x = np.zeros(dim)
                for d in range(dim):
                    stratum = rng.integers(10)
                    low_s = lo[d] + (hi[d] - lo[d]) * stratum / 10.0
                    high_s = lo[d] + (hi[d] - lo[d]) * (stratum + 1) / 10.0
                    new_x[d] = low_s + (high_s - low_s) * rng.random()
                new_f = problem(new_x)
                used += 1
                X[idx] = new_x
                F[idx] = new_f
                archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)

            # If stagnating, do a larger perturbation
            if stagnation_counter > 20 and used < budget - 10:
                idx = rng.integers(pop_size)
                # Larger step: combine archive point with random point
                if len(archive_X) > 0:
                    arch_idx = rng.integers(len(archive_X))
                    rand_idx = rng.integers(pop_size)
                    new_x = 0.5 * archive_X[arch_idx] + 0.5 * X[rand_idx]
                    new_x = np.clip(new_x + rng.normal(0, 0.3, dim), lo, hi)
                else:
                    new_x = lo + (hi - lo) * rng.random(dim)
                new_f = problem(new_x)
                used += 1
                X[idx] = new_x
                F[idx] = new_f
                archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)
                stagnation_counter = 0

        # ---------- Final local refinement: simple coordinate descent on top archive members ----------
        # Instead of expensive Nelder-Mead, use lightweight coordinate descent
        if len(archive_X) > 0 and time.time() - start_time < deadline - 10 and used < budget - 5:
            # Sort archive by crowding distance, take top few for polishing
            cd = crowding_distance(archive_F)
            order = np.argsort(-cd)  # descending crowding distance
            n_polish = min(6, len(archive_X))
            polish_indices = order[:n_polish]

            for pi in polish_indices:
                if used >= budget - 2 or time.time() - start_time > deadline - 5:
                    break

                x0 = archive_X[pi].copy()
                f0 = archive_F[pi].copy()
                
                # Simple coordinate descent: try small perturbations in each dimension
                for _ in range(3):  # 3 rounds of coordinate descent
                    if used >= budget - 1 or time.time() - start_time > deadline - 3:
                        break
                    for d in range(dim):
                        if used >= budget - 1 or time.time() - start_time > deadline - 3:
                            break
                        # Try small step in positive and negative direction
                        for direction in [1, -1]:
                            if used >= budget - 1 or time.time() - start_time > deadline - 3:
                                break
                            step_size = (hi[d] - lo[d]) * 0.05
                            trial = x0.copy()
                            trial[d] += direction * step_size
                            trial = np.clip(trial, lo, hi)
                            ft = problem(trial)
                            used += 1
                            # Accept if it doesn't dominate and is not dominated
                            if not is_dominated(ft, archive_F):
                                x0 = trial
                                f0 = ft
                                archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)
                                break  # Move to next dimension if improved

        return X
# EVOLVE-BLOCK-END
