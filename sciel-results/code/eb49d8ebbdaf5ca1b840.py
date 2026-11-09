import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using hypervolume-driven adaptive
    sampling with a final local-polish pass. Points are proposed by maximizing
    the expected marginal hypervolume contribution estimated from the current
    non-dominated set, then a short local refinement pass tightens the front."""

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

        def hv_contribution(f_new, F_arch, ref):
            """Estimate the marginal hypervolume contribution of f_new relative to F_arch."""
            if len(F_arch) == 0:
                # Contribution is the box from f_new to ref
                if np.all(f_new < ref):
                    return np.prod(ref - f_new)
                return 0.0
            # Simple approximation: contribution is the volume of the region
            # dominated by f_new but not by any point in F_arch.
            # We approximate by checking if f_new is non-dominated and
            # estimating the volume it adds.
            # For efficiency, we use a grid-based approximation in objective space.
            # But for small archives, we can do exact for 3D.
            # Here we use a simple heuristic: if f_new is non-dominated,
            # estimate contribution as the product of gaps to nearest neighbors.
            if np.any(f_new >= ref):
                return 0.0
            # Check if dominated
            for f_a in F_arch:
                if dominates(f_a, f_new):
                    return 0.0
            # Estimate contribution using nearest neighbors in each objective
            contrib = 1.0
            for m in range(n_obj):
                # Find the nearest archive point that is better in this objective
                better = F_arch[:, m] < f_new[m]
                if np.any(better):
                    nearest = np.min(F_arch[better, m])
                    gap = f_new[m] - nearest
                else:
                    gap = ref[m] - f_new[m]
                contrib *= max(gap, 0.0)
            return contrib

        # Phase 1: Initial diverse sampling
        pop_size = 30
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

        # Phase 2: Hypervolume-driven adaptive sampling
        while evals < budget and time.time() < deadline:
            if len(archive_F) == 0:
                # No archive yet, sample randomly
                x = rng.uniform(lo, hi)
                f = np.array(problem(x), dtype=float)
                evals += 1
                archive_X = np.array([x])
                archive_F = np.array([f])
                continue

            # Generate candidate proposals and pick the one with highest
            # estimated hypervolume contribution.
            n_candidates = 10
            best_contrib = -1.0
            best_x = None
            best_f = None

            for _ in range(n_candidates):
                # Propose a point: either DE-like from archive or random
                if len(archive_X) >= 3 and rng.random() < 0.7:
                    idxs = rng.choice(len(archive_X), size=3, replace=False)
                    X1, X2, X3 = archive_X[idxs[0]], archive_X[idxs[1]], archive_X[idxs[2]]
                    F_de = 0.5 + 0.3 * rng.random()
                    x_cand = X1 + F_de * (X2 - X3)
                else:
                    x_cand = rng.uniform(lo, hi)

                x_cand = np.clip(x_cand, lo, hi)

                # We can't evaluate f_cand without spending an eval, so we
                # estimate its objective values from a simple surrogate:
                # use the average of the archive points closest in decision space.
                if len(archive_X) > 0:
                    dists = np.linalg.norm(archive_X - x_cand, axis=1)
                    k = min(3, len(archive_X))
                    nearest_idx = np.argsort(dists)[:k]
                    f_est = np.mean(archive_F[nearest_idx], axis=0)
                else:
                    f_est = np.mean(ref) * np.ones(n_obj)

                contrib = hv_contribution(f_est, archive_F, ref)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_x = x_cand
                    best_f = f_est

            if best_x is None:
                break

            # Evaluate the best candidate
            f_actual = np.array(problem(best_x), dtype=float)
            evals += 1

            # Update archive
            if np.all(f_actual < ref):
                is_dom = any(dominates(af, f_actual) for af in archive_F)
                if not is_dom:
                    new_ax = []
                    new_af = []
                    for ax, af in zip(archive_X, archive_F):
                        if not dominates(f_actual, af):
                            new_ax.append(ax)
                            new_af.append(af)
                    new_ax.append(best_x)
                    new_af.append(f_actual)
                    archive_X = np.array(new_ax)
                    archive_F = np.array(new_af)

                    # Periodically prune archive if it gets too large
                    if len(archive_X) > 60:
                        nd_idx = non_dominated_indices(archive_F)
                        if len(nd_idx) > 40:
                            cd = crowding_distance(archive_F, nd_idx)
                            keep_order = np.argsort(-cd)[:40]
                            new_idx = nd_idx[keep_order]
                            archive_X = archive_X[new_idx]
                            archive_F = archive_F[new_idx]

        # Phase 3: Local refinement around best archive points
        remaining = budget - evals
        if remaining > 0 and len(archive_X) > 0 and time.time() < deadline:
            n_refine = min(remaining, 40)
            for _ in range(n_refine):
                if evals >= budget or time.time() > deadline:
                    break
                # Pick a random archive point and perturb
                idx = rng.integers(len(archive_X))
                base = archive_X[idx]
                sigma = 0.03 * (1.0 - evals / budget) + 0.01
                x_new = base + rng.normal(0, sigma, size=dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = np.array(problem(x_new), dtype=float)
                evals += 1

                # Update archive if non-dominated
                if np.all(f_new < ref):
                    is_dom = any(dominates(af, f_new) for af in archive_F)
                    if not is_dom:
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
