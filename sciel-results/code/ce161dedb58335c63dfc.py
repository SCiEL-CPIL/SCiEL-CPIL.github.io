# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using structured exploration and local refinement."""

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

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def _halton_sequence(self, n, dim, lo, hi):
        """Generate n Halton sequence points in [lo, hi]."""
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

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 15

        # Phase 1: Structured initialization with Halton + corners
        n_initial = min(67, budget // 3)

        # Halton points for space-filling coverage
        X_halton = self._halton_sequence(n_initial, dim, lo, hi)

        # Add structured corner and face-center points
        structured = []
        for i in range(2**dim):
            x = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            structured.append(x)
        # Face centers
        mid = (lo + hi) / 2
        for j in range(dim):
            x = mid.copy()
            x[j] = lo[j]
            structured.append(x.copy())
            x[j] = hi[j]
            structured.append(x.copy())
        structured.append(mid.copy())
        structured = np.array(structured)

        # Combine: use Halton, replace some with structured points
        n_structured = min(len(structured), n_initial // 3)
        if n_structured > 0:
            idx_replace = rng.choice(n_initial, size=n_structured, replace=False)
            X_init = X_halton.copy()
            X_init[idx_replace] = structured[:n_structured]
        else:
            X_init = X_halton.copy()

        F = np.array([problem(x) for x in X_init])
        used = len(X_init)
        all_F = F.copy()

        # Phase 2: Evolutionary search with adaptive mutation
        search_budget = int(budget * 0.65)

        while used < search_budget:
            if is_deadline():
                break

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)

            progress = used / max(search_budget, 1)
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

                # Replacement: find worst point in population
                nd_mask_pop = self._non_dominated_mask(F)
                dominated_idx = np.where(~nd_mask_pop)[0]

                if len(dominated_idx) > 0:
                    worst_idx = dominated_idx[rng.integers(len(dominated_idx))]
                else:
                    F_min = F.min(axis=0)
                    F_max = F.max(axis=0)
                    F_range = F_max - F_min
                    F_range[F_range == 0] = 1.0
                    F_norm = (F - F_min) / F_range
                    dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
                    worst_idx = np.argmax(dist)

                child_dominates_worst = np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx])
                worst_dominates_child = np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc)

                if child_dominates_worst:
                    X_init[worst_idx] = child
                    F[worst_idx] = fc
                elif not worst_dominates_child and np.any(fc < F[worst_idx]):
                    X_init[worst_idx] = child
                    F[worst_idx] = fc

        # Phase 3: Aggressive local refinement with weighted scalarization
        if used < budget and time_left() > 25:
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Generate systematic weight vectors for 3 objectives
                weights_list = []
                for a in np.linspace(0.05, 0.9, 6):
                    for b in np.linspace(0.05, 0.9 - a, 5):
                        c = 1.0 - a - b
                        if c >= 0.02:
                            weights_list.append([a, b, c])
                # Add extreme weights
                weights_list.append([0.95, 0.025, 0.025])
                weights_list.append([0.025, 0.95, 0.025])
                weights_list.append([0.025, 0.025, 0.95])
                # Add uniform
                weights_list.append([1/3, 1/3, 1/3])

                W = np.array(weights_list)
                # Normalize
                W = W / W.sum(axis=1, keepdims=True)

                # Limit weights by remaining budget
                max_weights = min(len(W), (budget - used) // 3)
                W = W[:max_weights]

                # Refine top non-dominated points
                points_to_refine = nd_idx[:min(6, len(nd_idx))]

                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 20:
                        break

                    x0 = X_init[pt_idx].copy()

                    for w in W:
                        if used >= budget or time_left() < 20:
                            break

                        def obj(x, w=w):
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 25, 'ftol': 1e-10, 'gtol': 1e-8})
                            if res.success or res.status in [1, 2]:
                                fx = problem(res.x)
                                used += 1
                                all_F = np.vstack([all_F, fx])

                                if np.any(fx < F[pt_idx]):
                                    X_init[pt_idx] = res.x
                                    F[pt_idx] = fx
                                    x0 = res.x.copy()
                        except:
                            pass

        # Phase 4: Final polish pass with tighter tolerance on best points
        if used < budget and time_left() > 15:
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Polish the best 3 points with very tight tolerance
                top_idx = nd_idx[:min(3, len(nd_idx))]
                for pt_idx in top_idx:
                    if used >= budget or time_left() < 10:
                        break

                    x0 = X_init[pt_idx].copy()
                    # Try a few different weight vectors for polishing
                    polish_weights = [
                        [1/3, 1/3, 1/3],
                        [0.5, 0.25, 0.25],
                        [0.25, 0.5, 0.25],
                        [0.25, 0.25, 0.5],
                    ]

                    for w in polish_weights:
                        if used >= budget or time_left() < 10:
                            break

                        def obj(x, w=w):
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10})
                            if res.success or res.status in [1, 2]:
                                fx = problem(res.x)
                                used += 1
                                all_F = np.vstack([all_F, fx])

                                if np.any(fx < F[pt_idx]):
                                    X_init[pt_idx] = res.x
                                    F[pt_idx] = fx
                                    x0 = res.x.copy()
                        except:
                            pass

        return X_init

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point in the population."""
        if len(F) == 0:
            return 0

        nd_mask = self._non_dominated_mask(F)

        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range

        dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
        penalty = np.where(~nd_mask, 50.0, 0.0)

        quality = dist + penalty
        return np.argmax(quality)
# EVOLVE-BLOCK-END
