# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using evolutionary search + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.array([True])

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)
        lt = np.any(F_i < F_j, axis=2)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)
        return ~dominated

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume contribution for a set of points."""
        if len(points) == 0:
            return 0.0
        # Sort by first coordinate
        idx = np.argsort(points[:, 0])
        points = points[idx]
        hv = 0.0
        for i in range(len(points)):
            w = ref[0] - points[i, 0]
            if w <= 0:
                continue
            h = ref[1] - points[i, 1]
            if h <= 0:
                continue
            # Width is limited by next point's first coordinate
            if i + 1 < len(points):
                w = min(w, points[i + 1, 0] - points[i, 0])
            hv += w * h
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 20

        # Phase 1: Structured initialization using Sobol-like quasi-random + corners
        n_initial = min(60, budget // 3)

        # Generate quasi-random points (Halton-like)
        def halton_sequence(n, dim, lo, hi):
            X = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11 for 5 dims
                vals = np.zeros(n)
                for i in range(n):
                    f = 1.0
                    r = i + 1
                    while r > 0:
                        f /= base
                        r = (r - 1) // base
                        vals[i] += f * ((r + 1) % base)
                X[:, d] = lo[d] + (hi[d] - lo[d]) * vals
            return X

        X_halton = halton_sequence(n_initial, dim, lo, hi)

        # Add corner points and midpoints
        structured = []
        for i in range(2**dim):
            x = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            structured.append(x)
        # Add face centers
        for j in range(dim):
            x = (lo + hi) / 2
            x[j] = lo[j]
            structured.append(x.copy())
            x[j] = hi[j]
            structured.append(x.copy())
        structured.append((lo + hi) / 2)
        structured = np.array(structured)

        # Combine: use Halton points, replace some with structured
        n_structured = min(len(structured), n_initial // 3)
        if n_structured > 0:
            rng = np.random.default_rng(42)
            idx_replace = rng.choice(n_initial, size=n_structured, replace=False)
            X_init = X_halton.copy()
            X_init[idx_replace] = structured[:n_structured]
        else:
            X_init = X_halton

        F = np.array([problem(x) for x in X_init])
        used = len(X_init)
        all_F = F.copy()

        # Phase 2: Evolutionary search with adaptive mutation
        pop_size = len(X_init)
        search_budget = int(budget * 0.65)

        rng = np.random.default_rng(123)

        while used < search_budget:
            if is_deadline():
                break

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)

            progress = used / max(search_budget, 1)
            # Adaptive mutation: start large, shrink over time
            mut_scale = 0.25 * (0.997 ** used)
            mut_scale = max(mut_scale, 0.02)

            n_children = min(15, search_budget - used)
            for _ in range(n_children):
                if is_deadline():
                    break
                if used >= search_budget:
                    break

                # Parent selection: tournament from non-dominated set
                if len(nd_idx) > 1 and rng.random() < 0.85:
                    t_size = min(3, len(nd_idx))
                    candidates = nd_idx[rng.choice(len(nd_idx), size=t_size, replace=False)]
                    F_cand = F[candidates]
                    ideal = F_cand.min(axis=0)
                    dists = np.sqrt(np.sum((F_cand - ideal) ** 2, axis=1))
                    parent_idx = candidates[np.argmin(dists)]
                else:
                    parent_idx = rng.integers(len(F))

                parent = X_init[parent_idx].copy()

                # Mutation: Gaussian with occasional larger jumps
                if rng.random() < 0.10:
                    jump_scale = mut_scale * 5.0
                else:
                    jump_scale = mut_scale

                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc])

                # Replacement strategy
                nd_mask_pop = self._non_dominated_mask(F)
                dominated_idx = np.where(~nd_mask_pop)[0]

                if len(dominated_idx) > 0:
                    worst_idx = dominated_idx[rng.integers(len(dominated_idx))]
                else:
                    ideal = F.min(axis=0)
                    dists = np.sqrt(np.sum((F - ideal) ** 2, axis=1))
                    worst_idx = np.argmax(dists)

                child_dominates_worst = np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx])
                worst_dominates_child = np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc)

                if child_dominates_worst:
                    X_init[worst_idx] = child
                    F[worst_idx] = fc
                elif not worst_dominates_child:
                    # Neither dominates: use simple diversity check
                    # Keep child if it improves spread
                    if np.any(fc < F[worst_idx]):
                        X_init[worst_idx] = child
                        F[worst_idx] = fc

        # Phase 3: Aggressive local refinement with weighted scalarization
        if used < budget and time_left() > 25:
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Generate diverse weight vectors
                n_weights = min(20, max(6, (budget - used) // 4))
                # Use a grid of weight vectors for better coverage
                W = np.zeros((n_weights, n_obj))
                if n_obj == 3:
                    # Systematic weight vectors for 3 objectives
                    weights_list = []
                    for a in np.linspace(0.1, 0.8, 5):
                        for b in np.linspace(0.1, 0.8 - a, 3):
                            c = 1.0 - a - b
                            if c > 0.05:
                                weights_list.append([a, b, c])
                    # Add some random weights
                    for _ in range(5):
                        w = rng.random(n_obj)
                        w /= w.sum()
                        weights_list.append(w)
                    # Also add extreme weights
                    weights_list.append([0.8, 0.1, 0.1])
                    weights_list.append([0.1, 0.8, 0.1])
                    weights_list.append([0.1, 0.1, 0.8])
                    weights_list.append([1/3, 1/3, 1/3])
                    
                    # Select n_weights from the list
                    if len(weights_list) >= n_weights:
                        selected = weights_list[:n_weights]
                    else:
                        selected = weights_list
                    W = np.array(selected)
                else:
                    W = rng.random((n_weights, n_obj))
                    W /= W.sum(axis=1, keepdims=True)

                # Sort non-dominated points by diversity
                F_nd = F[nd_idx]
                F_min = F_nd.min(axis=0)
                F_max = F_nd.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F_nd - F_min) / F_range
                dists = np.sqrt(np.sum(F_norm ** 2, axis=1))
                sorted_nd = nd_idx[np.argsort(dists)]

                # Refine top points with multiple weight vectors
                points_to_refine = sorted_nd[:min(12, len(sorted_nd))]
                
                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 25:
                        break

                    x0 = X_init[pt_idx].copy()

                    for w in W:
                        if used >= budget or time_left() < 25:
                            break

                        eval_count = [0]

                        def obj(x, w=w):
                            eval_count[0] += 1
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-13})
                            if eval_count[0] > 0:
                                fx = problem(res.x)
                                used += eval_count[0] + 1
                                all_F = np.vstack([all_F, fx])

                                # Update if improved in any objective
                                if np.any(fx < F[pt_idx]):
                                    X_init[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except Exception:
                            used += eval_count[0]

        return X_init
# EVOLVE-BLOCK-END
