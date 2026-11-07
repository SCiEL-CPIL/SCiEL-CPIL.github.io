# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: HV-guided local search with LHS init."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """2D hypervolume. pts: (m,2) array, all < ref. ref: (2,) array."""
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(m):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < m else ref[0]
            y_min = np.min(pts[:i + 1, 1])
            hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        """3D hypervolume. pts: (n,3) array, all < ref. ref: (3,) array."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _is_pareto(self, F):
        """Return boolean mask of Pareto-optimal points in F (n x 3)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            self_mask = np.ones(n, dtype=bool)
            self_mask[i] = False
            dominated = np.any(np.all(F[self_mask] <= F[i], axis=1) & np.any(F[self_mask] < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def _hv_contributions_batch(self, pareto_F, ref):
        """Compute HV contribution for each point in Pareto set."""
        n = len(pareto_F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([float(np.prod(ref - pareto_F[0]))])
        
        mask = np.all(pareto_F < ref, axis=1)
        valid_F = pareto_F[mask]
        if len(valid_F) == 0:
            return np.zeros(n)
        total_hv = self._hv_3d(valid_F, ref)
        
        contribs = np.zeros(n)
        for i in range(n):
            if not mask[i]:
                contribs[i] = 0.0
                continue
            rem_idx = [j for j in range(n) if j != i and mask[j]]
            if len(rem_idx) == 0:
                contribs[i] = total_hv
            else:
                rem_F = pareto_F[rem_idx]
                hv_rem = self._hv_3d(rem_F, ref)
                contribs[i] = max(total_hv - hv_rem, 1e-10)
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        # Use global np.random (seeded by evaluator) for reproducibility
        rng = np.random  # Use the global seeded RNG

        # --- Phase 1: Initial sampling ---
        n_lhs = 40
        n_rand = 20
        n_init = min(n_lhs + n_rand, budget)

        X_list = []
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)
        X_list.append(lhs_X)

        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        X_list.append(rand_X)

        X_init = np.vstack(X_list)
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        all_X = X_init.copy()
        all_F = F_init.copy()

        # Build initial Pareto archive
        valid = np.all(F_init < ref, axis=1)
        if np.any(valid):
            pF = F_init[valid]
            pX = X_init[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # Cache HV contributions
        cached_contribs = None
        pareto_version = 0

        # --- Phase 2: Iterative improvement ---
        polish_start = int(budget * 0.75)  # Last 25% for polishing

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            is_polish = used >= polish_start

            # Parent selection
            if len(pareto_F) > 0:
                # Recompute HV contributions only when Pareto set changed
                if cached_contribs is None or len(cached_contribs) != len(pareto_F):
                    cached_contribs = self._hv_contributions_batch(pareto_F, ref)
                    pareto_version += 1

                contribs = np.clip(cached_contribs, 1e-10, None)
                probs = contribs / contribs.sum()

                # Temperature annealing
                temp = 2.0 * (1.0 - progress) + 0.5
                if is_polish:
                    temp = 0.8  # Lower temperature for exploitation
                weighted = np.power(contribs, 1.0 / temp)
                probs_w = weighted / weighted.sum()
                parent_idx = np.random.choice(len(pareto_F), p=probs_w)
                parent = pareto_X[parent_idx]
            else:
                if len(all_X) > 0:
                    parent = all_X[np.random.randint(len(all_X))]
                else:
                    parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Mutation strategy: adaptive step size
            if is_polish:
                # Smaller steps for polishing
                base_step = 0.05 + 0.15 * (1.0 - (used - polish_start) / max(1, budget - polish_start))
                step = base_step * (0.5 + np.random.uniform())
                # Mostly coordinate-specific mutations
                use_coord = np.random.uniform() < 0.8
            else:
                base_step = 0.8 * (1.0 - progress * 0.7) + 0.1
                step = base_step * (0.5 + np.random.uniform())
                use_coord = np.random.uniform() < 0.3

            if not use_coord:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                n_perturb = np.random.randint(1, 3)
                coords = np.random.choice(dim, size=n_perturb, replace=False)
                for c in coords:
                    child[c] = parent[c] + step * (hi[c] - lo[c]) * np.random.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X = np.vstack([all_X, [child]])
            all_F = np.vstack([all_F, [fc]])

            # Update Pareto archive
            if np.all(fc < ref):
                dominated = False
                for pf in pareto_F:
                    if np.all(pf <= fc) and np.any(pf < fc):
                        dominated = True
                        break

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    # Remove points dominated by the new point
                    keep = np.ones(len(pareto_F), dtype=bool)
                    for i in range(len(pareto_F) - 1):
                        if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                            keep[i] = False
                    pareto_X = pareto_X[keep]
                    pareto_F = pareto_F[keep]
                    cached_contribs = None  # Invalidate cache

                # Prune archive if too large
                if len(pareto_F) > 50:
                    p_mask = self._is_pareto(pareto_F)
                    pareto_X = pareto_X[p_mask]
                    pareto_F = pareto_F[p_mask]
                    if len(pareto_F) > 50:
                        contribs = self._hv_contributions_batch(pareto_F, ref)
                        top_idx = np.argsort(contribs)[-50:]
                        pareto_X = pareto_X[top_idx]
                        pareto_F = pareto_F[top_idx]
                    cached_contribs = None

        # --- Phase 3: Final polish with local optimization ---
        if used < budget and len(pareto_F) > 0:
            n_polish = min(budget - used, 30)
            # Pick top 5 Pareto points by HV contribution and do local refinement
            contribs = self._hv_contributions_batch(pareto_F, ref)
            top_k = min(5, len(pareto_F))
            top_idx = np.argsort(contribs)[-top_k:]
            
            for pi in top_idx:
                if used >= budget or time.time() > deadline:
                    break
                parent = pareto_X[pi]
                # Do 5-6 local steps around this point
                for _ in range(6):
                    if used >= budget or time.time() > deadline:
                        break
                    step = 0.02 * np.random.uniform()
                    child = parent + step * (hi - lo) * np.random.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    all_X = np.vstack([all_X, [child]])
                    all_F = np.vstack([all_F, [fc]])
                    if np.all(fc < ref):
                        dominated = False
                        for pf in pareto_F:
                            if np.all(pf <= fc) and np.any(pf < fc):
                                dominated = True
                                break
                        if not dominated:
                            pareto_X = np.vstack([pareto_X, [child]])
                            pareto_F = np.vstack([pareto_F, [fc]])
                            keep = np.ones(len(pareto_F), dtype=bool)
                            for i in range(len(pareto_F) - 1):
                                if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                                    keep[i] = False
                            pareto_X = pareto_X[keep]
                            pareto_F = pareto_F[keep]
                            cached_contribs = None

        return all_X

# EVOLVE-BLOCK-END
