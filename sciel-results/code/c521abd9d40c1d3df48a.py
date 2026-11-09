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
        deadline = start + 275.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
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
            return mask

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

        def hv_contribution(point, others, ref):
            """Estimate hypervolume contribution of a point relative to others."""
            # For 3 objectives, a point contributes if it's not dominated and
            # we can estimate its unique contribution
            if np.any(point >= ref):
                return 0.0
            # Simple estimate: volume of the box from point to ref,
            # minus overlap with other points
            vol = np.prod(ref - point)
            # Discount by number of points that are close in all objectives
            discount = 1.0
            for o in others:
                if np.all(o <= point) or np.all(o >= point):
                    continue
                # If o is close, discount
                dist = np.sum(np.abs(o - point)) / np.sum(ref)
                discount *= (1.0 - 0.3 * dist)
            return vol * max(discount, 0.1)

        # Phase 1: Initial diverse sampling
        pop_size = 40
        X = []
        F = []
        evals = 0

        # Latin hypercube-style initialization for diversity
        for i in range(pop_size):
            if evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            evals += 1

        X = np.array(X)
        F = np.array(F)

        # Build initial archive (non-dominated set)
        nd_mask = non_dominated_mask(F)
        archive_idx = np.where(nd_mask)[0]
        archive_X = X[archive_idx]
        archive_F = F[archive_idx]

        best_hv = 0.0

        # Phase 2: Main search loop with adaptive strategy
        generation = 0
        F_de = 0.7
        CR = 0.8

        while evals < budget and time.time() < deadline:
            generation += 1

            # Adaptive F_de: decrease over time for exploitation
            progress = evals / budget
            F_de = 0.8 - 0.4 * progress
            F_de = max(F_de, 0.3)

            # Generate offspring using DE-like mutation from archive
            n_offspring = min(pop_size, budget - evals)
            if n_offspring <= 0:
                break

            offspring_X = []
            offspring_F = []

            for k in range(n_offspring):
                if evals >= budget or time.time() > deadline:
                    break

                # Select parents from archive
                n_arch = len(archive_X)
                if n_arch >= 3:
                    # Weighted selection: prefer points with higher HV contribution
                    weights = np.ones(n_arch)
                    ref = np.array([1.1, 1.3, 1.1])
                    for i in range(n_arch):
                        others = [archive_F[j] for j in range(n_arch) if j != i]
                        weights[i] = hv_contribution(archive_F[i], others, ref)
                    weights = weights / weights.sum()

                    # Select 3 distinct parents
                    while True:
                        idxs = rng.choice(n_arch, size=3, replace=False, p=weights)
                        if len(set(idxs)) == 3:
                            break
                    X1, X2, X3 = archive_X[idxs[0]], archive_X[idxs[1]], archive_X[idxs[2]]
                else:
                    X1 = rng.uniform(lo, hi)
                    X2 = rng.uniform(lo, hi)
                    X3 = rng.uniform(lo, hi)

                # DE mutation
                v = X1 + F_de * (X2 - X3)

                # Crossover (binomial)
                trial = v.copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X1[j]

                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = np.array(problem(trial), dtype=float)
                evals += 1

                offspring_X.append(trial)
                offspring_F.append(f_trial)

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Selection: combine current archive with offspring, select best
            # by non-dominated sorting + crowding distance
            all_X = np.vstack([archive_X, offspring_X])
            all_F = np.vstack([archive_F, offspring_F])

            # Find non-dominated
            nd_mask = non_dominated_mask(all_F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) > pop_size:
                # Keep top pop_size by crowding distance
                cd = crowding_distance(all_F, nd_indices)
                keep_order = np.argsort(-cd)
                keep_order = keep_order[:pop_size]
                new_archive_idx = nd_indices[keep_order]
            else:
                new_archive_idx = nd_indices

            archive_X = all_X[new_archive_idx]
            archive_F = all_F[new_archive_idx]

            # Also maintain a broader population for parent selection diversity
            # Every few generations, refresh population with random points
            if generation % 5 == 0 and evals < budget - 10:
                n_rand = min(5, budget - evals)
                for _ in range(n_rand):
                    if evals >= budget or time.time() > deadline:
                        break
                    xr = rng.uniform(lo, hi)
                    xr = np.clip(xr, lo, hi)
                    fr = np.array(problem(xr), dtype=float)
                    # Add to archive if non-dominated
                    if len(archive_F) == 0 or not any(dominates(af, fr) for af in archive_F):
                        # Remove any dominated by fr
                        new_ax = []
                        new_af = []
                        for ax, af in zip(archive_X, archive_F):
                            if not dominates(fr, af):
                                new_ax.append(ax)
                                new_af.append(af)
                        new_ax.append(xr)
                        new_af.append(fr)
                        archive_X = np.array(new_ax)
                        archive_F = np.array(new_af)
                    evals += 1

        return archive_X
