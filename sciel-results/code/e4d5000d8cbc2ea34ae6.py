"""Multi-objective optimizer for RE34 using iterative DE search with Pareto archive."""
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
        deadline = 310.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(0)

        # ---------- Initialization: structured diversity ----------
        # 1) All 2^5 = 32 corners
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )

        # 2) Midpoint and face centers
        mid = (lo + hi) / 2.0
        face_centers = []
        for d in range(dim):
            for val in [lo[d], hi[d]]:
                fc = mid.copy()
                fc[d] = val
                face_centers.append(fc.copy())

        pop_size = 30
        init_points = [corners[0].copy()]
        for i in [1, 2, 4, 8, 16]:
            init_points.append(corners[i].copy())
        init_points.append(mid.copy())
        # Add some face centers for better coverage
        for fc in face_centers[:4]:
            init_points.append(fc.copy())

        # Fill remaining with stratified random sampling
        n_remaining = pop_size - len(init_points)
        if n_remaining > 0:
            for _ in range(n_remaining):
                x = lo + (hi - lo) * rng.random(dim)
                init_points.append(x)

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

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

        # Track evaluation count for Nelder-Mead polish
        eval_count = [0]
        def counting_scalar_obj(x, weights):
            eval_count[0] += 1
            fx = problem(x)
            return np.sum(weights * fx)

        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Select parent from archive using crowding distance
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

            # Replace a random population member to maintain diversity
            # This avoids the O(N) scan and keeps the population diverse
            replace_idx = rng.integers(pop_size)
            X[replace_idx] = trial
            F[replace_idx] = ft

        # ---------- Final polish: local optimization on top archive members ----------
        # Use remaining budget for targeted local search on diverse archive points
        if len(archive_X) > 0 and time.time() - start_time < deadline - 5:
            cd = crowding_distance(archive_F)
            order = np.argsort(-cd)  # descending crowding distance
            n_polish = min(3, len(archive_X))
            polish_indices = order[:n_polish]

            arch_F_arr = np.array(archive_F)
            obj_min = arch_F_arr.min(axis=0)
            obj_max = arch_F_arr.max(axis=0)
            obj_range = obj_max - obj_min
            obj_range[obj_range == 0] = 1.0
            weights = 1.0 / obj_range

            for pi in polish_indices:
                if used >= budget or time.time() - start_time > deadline - 3:
                    break

                x0 = archive_X[pi].copy()
                eval_count[0] = 0

                try:
                    res = minimize(
                        lambda x: counting_scalar_obj(x, weights), x0,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
                    )
                    # Count all evaluations made by Nelder-Mead
                    used += eval_count[0]
                    if used <= budget:
                        f_res = problem(res.x)
                        archive_X, archive_F = update_archive(res.x, f_res, archive_X, archive_F)
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
