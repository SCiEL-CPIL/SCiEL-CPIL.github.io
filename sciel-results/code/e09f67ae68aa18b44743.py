import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using population-based search with
    hypervolume-based selection and adaptive exploration/exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)
        ref = np.array([1.1, 1.3, 1.1])

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices in F."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                span = vals[order[-1]] - vals[order[0]]
                if span > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span
            return cd

        # Phase 1: Initial diverse sampling
        pop_size = 50
        X = []
        F = []
        evals = 0

        for i in range(pop_size):
            if evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            evals += 1

        X = np.array(X)
        F = np.array(F)

        # Build initial archive
        nd_idx = non_dominated_indices(F)
        archive_X = X[nd_idx]
        archive_F = F[nd_idx]

        # Phase 2: Main search loop
        generation = 0
        while evals < budget and time.time() < deadline:
            generation += 1

            progress = evals / budget
            F_de = 0.8 - 0.5 * progress
            F_de = max(F_de, 0.2)
            CR = 0.8

            n_offspring = min(pop_size, budget - evals)
            if n_offspring <= 0:
                break

            offspring_X = []
            offspring_F = []

            n_arch = len(archive_X)
            for k in range(n_offspring):
                if evals >= budget or time.time() > deadline:
                    break

                if n_arch >= 3:
                    # Uniform random selection from archive (fast)
                    idxs = rng.choice(n_arch, size=3, replace=False)
                    X1, X2, X3 = archive_X[idxs[0]], archive_X[idxs[1]], archive_X[idxs[2]]
                else:
                    X1 = rng.uniform(lo, hi)
                    X2 = rng.uniform(lo, hi)
                    X3 = rng.uniform(lo, hi)

                # DE mutation
                v = X1 + F_de * (X2 - X3)

                # Binomial crossover
                trial = v.copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X1[j]

                trial = np.clip(trial, lo, hi)
                f_trial = np.array(problem(trial), dtype=float)
                evals += 1

                offspring_X.append(trial)
                offspring_F.append(f_trial)

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            all_X = np.vstack([archive_X, offspring_X])
            all_F = np.vstack([archive_F, offspring_F])

            nd_idx = non_dominated_indices(all_F)

            if len(nd_idx) > pop_size:
                cd = crowding_distance(all_F, nd_idx)
                keep_order = np.argsort(-cd)[:pop_size]
                new_idx = nd_idx[keep_order]
            else:
                new_idx = nd_idx

            archive_X = all_X[new_idx]
            archive_F = all_F[new_idx]

        # Phase 3: Local refinement around best archive points
        remaining = budget - evals
        if remaining > 0 and len(archive_X) > 0 and time.time() < deadline:
            n_refine = min(remaining, 30)
            for _ in range(n_refine):
                if evals >= budget or time.time() > deadline:
                    break
                # Pick a random archive point and perturb
                idx = rng.integers(len(archive_X))
                base = archive_X[idx]
                # Small random perturbation
                sigma = 0.05 * (1.0 - evals / budget)
                x_new = base + rng.normal(0, sigma, size=dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = np.array(problem(x_new), dtype=float)
                evals += 1

                # Update archive if non-dominated
                is_dom = any(dominates(af, f_new) for af in archive_F)
                if not is_dom:
                    # Remove dominated points
                    new_ax = []
                    new_af = []
                    for ax, af in zip(archive_X, archive_F):
                        if not dominates(f_new, af):
                            new_ax.append(ax)
                            new_af.append(af)
                    new_ax.append(x_new)
                    new_af.append(f_new)
                    archive_X = np.array(new_ax)
                    archive_F = np.array(new_af)

        return archive_X
