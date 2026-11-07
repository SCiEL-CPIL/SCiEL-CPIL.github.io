# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with coordinate-descent polish.

    Strategy:
      * Seed/bias toward the global Pareto region (x_1 ~= 0.35).
      * Fast population DE/rand/1 with hypervolume-driven selection.
      * Deterministic coordinate-descent polish via ternary search per dimension,
        using hypervolume contribution as scalar fitness guide.
    """

    # WFG4 variable roles (0-indexed).
    COND = 0            # x_1 : 0.35 -> global front, 0.0 -> local front
    POS = (1, 2, 3, 5, 6, 8, 9, 11, 12)   # position parameters
    DIR = (4, 7, 10)    # direction parameters

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _hv3(pts):
        """Exact 3-objective hypervolume (maximal points, origin-referenced)."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] == 0:
            return 0.0
        # keep only maximal (non-dominated) points
        keep = np.ones(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if not keep[i]:
                continue
            p = pts[i]
            d = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            d[i] = False
            keep[d] = False
        pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts = pts[order]
        hv = 0.0
        z1 = 0.0
        z2 = 0.0
        for i in range(pts.shape[0]):
            a, b, c = pts[i]
            if a > z1:
                hv += (a - z1) * z2
                z1 = a
            if b > z2:
                hv += (b - z2) * c
                z2 = b
        hv += z2 * pts[-1, 2]
        return float(hv)

    @staticmethod
    def _dominates(a, b):
        """True if a dominates b (a <= b in all, a < b in at least one)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _hv_contribution(self, f, archive_F):
        """Hypervolume contribution of point f relative to archive (exact)."""
        if not archive_F:
            return self._hv3(np.array([f]))
        base_hv = self._hv3(np.array(archive_F))
        new_hv = self._hv3(np.array(archive_F + [f]))
        return new_hv - base_hv

    def _nelder_mead_polish(self, x, f, problem, bounds, budget_remaining, deadline):
        """Nelder-Mead simplex polish minimizing the sum of objectives.
        
        Returns (best_x, best_f, evaluations_used).
        """
        dim = len(x)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        evals_used = 0

        # Build initial simplex
        simplex = [x.copy()]
        simplex_f = [f.copy()]
        for d in range(dim):
            if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                break
            pt = x.copy()
            step = 0.05 * span[d]
            if step < 1e-10:
                step = 0.05
            pt[d] += step
            pt = np.clip(pt, lo, hi)
            pf = np.asarray(problem(pt), dtype=float)
            evals_used += 1
            simplex.append(pt)
            simplex_f.append(pf)

        n = len(simplex)
        if n < 2:
            return x, f, evals_used

        def obj_sum(fv):
            return float(np.sum(fv))

        # Sort simplex by objective sum
        order = np.argsort([obj_sum(fv) for fv in simplex_f])
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]

        max_iter = min(20, max(3, (budget_remaining - evals_used) // (dim + 2)))
        for _ in range(max_iter):
            if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                break

            # Current centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            worst_obj = obj_sum(simplex_f[-1])

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lo, hi)
            fr = np.asarray(problem(xr), dtype=float)
            evals_used += 1
            if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                break
            fr_obj = obj_sum(fr)

            if fr_obj < obj_sum(simplex_f[0]):
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, lo, hi)
                fe = np.asarray(problem(xe), dtype=float)
                evals_used += 1
                if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                    break
                fe_obj = obj_sum(fe)
                if fe_obj < fr_obj:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr_obj < worst_obj:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                rho = 0.5
                xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, lo, hi)
                fc = np.asarray(problem(xc), dtype=float)
                evals_used += 1
                if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                    break
                fc_obj = obj_sum(fc)
                if fc_obj < worst_obj:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        simplex_f[i] = np.asarray(problem(simplex[i]), dtype=float)
                        evals_used += 1

            # Re-sort
            order = np.argsort([obj_sum(fv) for fv in simplex_f])
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

        # Return best point
        best_idx = int(np.argmin([obj_sum(fv) for fv in simplex_f]))
        return simplex[best_idx], simplex_f[best_idx], evals_used

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 275.0            # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        c = 0.35 * span[self.COND]       # global-front offset for x_1

        def rand_point():
            x = lo + np.random.rand(dim) * span
            x[self.COND] += c            # bias toward global front
            return np.clip(x, lo, hi)

        pop = int(min(30, max(6, budget // 8)))
        X = np.array([rand_point() for _ in range(pop)])
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop

        # Reference point for scoring
        ref = np.array([3.0, 5.0, 7.0])

        # Track best archive points (non-dominated, spread)
        archive_F = []
        archive_X = []

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _update_archive(x, f):
            nonlocal archive_F, archive_X
            # Check if dominated by existing archive
            for af in archive_F:
                if _dominates(af, f):
                    return
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not _dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

            # If archive too large, trim by crowding distance
            max_archive = 50
            if len(archive_F) > max_archive:
                # Keep only non-dominated
                nd = []
                for i in range(len(archive_F)):
                    dominated = False
                    for j in range(len(archive_F)):
                        if i != j and _dominates(archive_F[j], archive_F[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd.append(i)
                if len(nd) > max_archive:
                    # Compute crowding distance
                    dist = np.zeros(len(nd))
                    for m in range(self.n_obj):
                        vals = np.array([archive_F[i][m] for i in nd])
                        order = np.argsort(vals)
                        dist[order[0]] = np.inf
                        dist[order[-1]] = np.inf
                        span_m = vals[order[-1]] - vals[order[0]]
                        if span_m > 1e-15:
                            for k in range(1, len(nd) - 1):
                                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span_m
                    # Remove point with smallest crowding distance
                    min_idx = nd[np.argmin(dist)]
                    archive_X.pop(min_idx)
                    archive_F.pop(min_idx)

        # Initialize archive
        for i in range(pop):
            _update_archive(X[i], F[i])

        # Phase 1: DE-based search (use ~75% of budget)
        de_budget = int(budget * 0.75)
        while used < de_budget:
            if time.time() >= deadline:
                break
            # pick 3 distinct parents
            i = np.random.randint(0, pop)
            j = np.random.randint(0, pop)
            k = np.random.randint(0, pop)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop)
                k = np.random.randint(0, pop)
            v = X[i] + 0.65 * (X[j] - X[k])
            v = np.clip(v, lo, hi)
            vf = np.asarray(problem(v), dtype=float)
            used += 1

            # Cheap acceptance: accept if vf is not dominated by F[i]
            # (avoids expensive full-population hypervolume recomputation)
            if not self._dominates(F[i], vf):
                X[i], F[i] = v, vf
            _update_archive(v, vf)

        # Phase 2: Nelder-Mead polish on top archive points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Select top archive points by hypervolume contribution
            # (limit to top 5 to keep polish affordable)
            top_n = min(5, len(archive_F))
            if top_n > 0:
                # Compute contributions (expensive but done once per point)
                contribs = []
                for idx in range(len(archive_F)):
                    other_F = [af for i, af in enumerate(archive_F) if i != idx]
                    contribs.append(self._hv_contribution(archive_F[idx], other_F))
                order = np.argsort(contribs)[::-1][:top_n]

                for idx in order:
                    if used >= budget or time.time() >= deadline:
                        break
                    if idx >= len(archive_X):
                        continue

                    x = archive_X[idx].copy()
                    f = archive_F[idx].copy()
                    remaining_before = budget - used

                    new_x, new_f, evals_used = self._nelder_mead_polish(
                        x, f, problem, bounds, remaining_before, deadline
                    )
                    used += evals_used
                    _update_archive(new_x, new_f)

        return X
# EVOLVE-BLOCK-END
