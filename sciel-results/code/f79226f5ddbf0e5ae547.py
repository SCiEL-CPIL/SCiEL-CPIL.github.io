"""Multi-objective optimizer on RE37 with deadline guard and scalarization-based local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Weighted-sum scalarization + local search + non-dominated archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 285.0  # safety margin before 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj

        used = 0
        max_evals = self.budget

        # Non-dominated archive
        archive_X = []
        archive_F = []

        def dominated(f, archive_F):
            # True if f is dominated by any point in archive_F
            if len(archive_F) == 0:
                return False
            A = np.asarray(archive_F)
            # f dominated by a: a <= f and a < f in at least one
            return np.any(np.all(A <= f, axis=1) & np.any(A < f, axis=1))

        def is_pareto(f, archive_F):
            # True if f is not dominated by any archive point
            if len(archive_F) == 0:
                return True
            A = np.asarray(archive_F)
            return not np.any(np.all(A <= f, axis=1) & np.any(A < f, axis=1))

        def add_to_archive(x, f):
            nonlocal used
            # Remove archive points dominated by f
            new_X = []
            new_F = []
            for xi, fi in zip(archive_X, archive_F):
                # keep xi if not dominated by f
                if not (np.all(f <= fi) and np.any(f < fi)):
                    new_X.append(xi)
                    new_F.append(fi)
            archive_X[:] = new_X
            archive_F[:] = new_F
            # Add f if not dominated by remaining archive
            if is_pareto(f, archive_F):
                archive_X.append(np.array(x, dtype=float))
                archive_F.append(np.array(f, dtype=float))

        def eval_point(x):
            nonlocal used
            if used >= max_evals:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        # Local search on scalarized objective from a start point
        def local_search_scalar(x0, w, max_iter=25):
            nonlocal used
            if used >= max_evals:
                return None
            x0 = np.clip(x0, lo, hi)
            # quick local Nelder-Mead on weighted sum
            used_local = 0
            def obj(x):
                nonlocal used
                if used >= max_evals:
                    return 1e9
                f = np.asarray(problem(x), dtype=float)
                used += 1
                used_local += 1
                return float(np.dot(w, f))
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-9,
                                        'adaptive': True})
                xbest = np.clip(res.x, lo, hi)
                fbest = np.asarray(problem(xbest), dtype=float)
                used += 1
                return xbest, fbest
            except Exception:
                return None

        # Phase 1: structured weight vectors (grid) + random
        # Generate weight vectors on a simplex-like grid for n_obj=3, dim=4
        n_weights = 12
        weight_vectors = []
        # uniform random weights
        for _ in range(n_weights):
            w = np.random.rand(n_obj)
            w = w / w.sum()
            weight_vectors.append(w)
        # Add canonical-ish weights to ensure coverage
        weight_vectors.append(np.array([1.0, 0.0, 0.0]))
        weight_vectors.append(np.array([0.0, 1.0, 0.0]))
        weight_vectors.append(np.array([0.0, 0.0, 1.0]))
        weight_vectors.append(np.array([0.5, 0.25, 0.25]))
        weight_vectors.append(np.array([0.25, 0.5, 0.25]))
        weight_vectors.append(np.array([0.25, 0.25, 0.5]))
        weight_vectors.append(np.array([1.0/3, 1.0/3, 1.0/3]))

        # Initial random points
        n_init = min(8, max_evals)
        for i in range(n_init):
            x = lo + span * np.random.rand(dim)
            res = eval_point(x)
            if res is None:
                break
            x, f = res
            add_to_archive(x, f)

        # Phase 2: for each weight vector, local search from a random start (or from archive)
        for w in weight_vectors:
            if used >= max_evals or time.time() > deadline:
                break
            # Start from a random point
            x0 = lo + span * np.random.rand(dim)
            res = local_search_scalar(x0, w, max_iter=20)
            if res is not None:
                x, f = res
                add_to_archive(x, f)

        # Phase 3: refine archive points with small random perturbations + local search
        # Iterate over a copy of the archive; add new non-dominated points
        max_refine_rounds = 3
        for rnd in range(max_refine_rounds):
            if used >= max_evals or time.time() > deadline:
                break
            # snapshot
            snap = list(archive_X)
            for x in snap:
                if used >= max_evals or time.time() > deadline:
                    break
                # random perturbation
                step = 0.1 * span
                x_p = x + step * np.random.randn(dim)
                x_p = np.clip(x_p, lo, hi)
                res = eval_point(x_p)
                if res is None:
                    continue
                xp, fp = res
                if is_pareto(fp, archive_F):
                    add_to_archive(xp, fp)
                    # local search from this new point with a random weight
                    w = np.random.rand(n_obj)
                    w = w / w.sum()
                    res2 = local_search_scalar(xp, w, max_iter=15)
                    if res2 is not None:
                        x2, f2 = res2
                        add_to_archive(x2, f2)

        # Phase 4: if budget remains, do more random scalarized local searches
        while used < max_evals and time.time() < deadline:
            w = np.random.rand(n_obj)
            w = w / w.sum()
            x0 = lo + span * np.random.rand(dim)
            res = local_search_scalar(x0, w, max_iter=15)
            if res is not None:
                x, f = res
                add_to_archive(x, f)

        # Ensure at least something returned
        if len(archive_X) == 0:
            x = lo + span * np.random.rand(dim)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            archive_X.append(np.array(x))
            archive_F.append(np.array(f))

        return np.array(archive_X)
# EVOLVE-BLOCK-END
