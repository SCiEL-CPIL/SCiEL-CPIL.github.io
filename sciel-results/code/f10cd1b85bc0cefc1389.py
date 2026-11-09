"""Multi-objective optimizer for RE34 using LHS initialization and DE search with Pareto archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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

        lh_samples = np.zeros((n_lh, dim))
        for d in range(dim):
            strata = np.linspace(0, 1, n_lh + 1)
            for i in range(n_lh):
                lh_samples[i, d] = strata[i] + (strata[i + 1] - strata[i]) * rng.random()
        
        X_init = lo + (hi - lo) * lh_samples

        # Add corners for boundary exploration
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )
        corner_indices = [0, 1, 2, 4, 8, 16]
        for i, ci in enumerate(corner_indices):
            X_init[i] = corners[ci]

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
        F_factor = 0.8
        CR = 0.9

        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Select parent from archive using tournament on crowding distance
            if len(archive_X) > 1:
                cd = crowding_distance(archive_F)
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

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Update population: replace worst member by non-dominated count
            # Find dominated points in population
            dom_count = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and dominates(F[i], F[j]):
                        dom_count[j] += 1
            worst_idx = np.argmax(dom_count)
            if dom_count[worst_idx] == 0:
                # All non-dominated among themselves; use sum as tiebreaker
                worst_idx = np.argmax(np.sum(F, axis=1))
            
            if not dominates(F[worst_idx], ft):
                X[worst_idx] = trial
                F[worst_idx] = ft
            else:
                if rng.random() < 0.15:
                    X[worst_idx] = trial
                    F[worst_idx] = ft

            # Periodic diversity injection: replace one random population member
            if used % 40 == 0 and used < budget - 5:
                idx = rng.integers(pop_size)
                new_x = lo + (hi - lo) * rng.random(dim)
                new_f = problem(new_x)
                used += 1
                X[idx] = new_x
                F[idx] = new_f
                archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)

        # ---------- Final polish: bounded local optimization on top archive members ----------
        remaining = budget - used
        if len(archive_X) > 0 and remaining > 3 and time.time() - start_time < deadline - 20:
            cd = crowding_distance(archive_F)
            order = np.argsort(-cd)
            n_polish = min(6, len(archive_X), remaining // 3)
            polish_indices = order[:n_polish]

            for pi in polish_indices:
                if used >= budget - 2 or time.time() - start_time > deadline - 15:
                    break

                x0 = archive_X[pi].copy()
                arch_F_arr = np.array(archive_F)
                obj_min = arch_F_arr.min(axis=0)
                obj_max = arch_F_arr.max(axis=0)
                obj_range = obj_max - obj_min
                obj_range[obj_range == 0] = 1.0
                weights = 1.0 / obj_range

                eval_count = [0]

                def scalar_obj(x):
                    fx = problem(np.clip(x, lo, hi))
                    eval_count[0] += 1
                    return np.sum(weights * fx)

                try:
                    res = minimize(
                        scalar_obj, x0,
                        method='Nelder-Mead',
                        options={'maxiter': 40, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if used + 1 <= budget:
                        f_res = problem(np.clip(res.x, lo, hi))
                        used += 1
                        archive_X, archive_F = update_archive(res.x, f_res, archive_X, archive_F)
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
