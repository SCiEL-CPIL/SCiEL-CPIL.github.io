# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: fast non-dominated sorting + crowding
    based evolutionary search with a time-budgeted local-search polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- helpers (kept for compatibility, not used in main loop) ----
    def _hv_2d(self, pts, ref):
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
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            self_mask = np.ones(n, dtype=bool)
            self_mask[i] = False
            dominated = np.any(np.all(F[self_mask] <= F[i], axis=1) &
                               np.any(F[self_mask] < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def _hv_contributions_batch(self, pareto_F, ref):
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
                continue
            rem_idx = [j for j in range(n) if j != i and mask[j]]
            if len(rem_idx) == 0:
                contribs[i] = total_hv
            else:
                hv_rem = self._hv_3d(pareto_F[rem_idx], ref)
                contribs[i] = max(total_hv - hv_rem, 1e-10)
        return contribs

    # ---- new fast machinery ----
    def _dominates(self, a, B):
        """Return bool array: does point a dominate each row of B?"""
        return np.all(B >= a, axis=1) & np.any(B > a, axis=1)

    def _is_dom(self, F):
        """True where F[i] is dominated by some other point in F."""
        n = len(F)
        if n <= 1:
            return np.zeros(n, dtype=bool)
        ge = F[:, None, :] >= F[None, :, :]      # (i,j): F[i]>=F[j]
        gt = F[:, None, :] > F[None, :, :]
        np.fill_diagonal(ge, False)
        np.fill_diagonal(gt, False)
        return np.any(ge & np.any(gt, axis=2), axis=1)

    def _fast_nondominated_sort(self, F):
        """Return list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        front0 = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                front0.append(i)
        fronts.append(front0)
        k = 0
        while len(fronts[k]) > 0:
            nxt = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        nxt.append(j)
            k += 1
            if nxt:
                fronts.append(nxt)
        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        cd = np.zeros(n)
        if n <= 2:
            return cd
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            rng = F[order[-1], m] - F[order[0], m]
            if rng == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / rng
        return cd

    def _local_search(self, x, problem, lo, hi, ref, budget_left, deadline):
        """Cheap local improvement around x. Returns improved point + count used."""
        x = np.clip(x.astype(np.float64), lo, hi).copy()
        fx = np.array(problem(x))
        used = 0
        dim = self.dim
        # small random-restart hill climb
        for _ in range(4):
            best = x.copy()
            bestf = fx.copy()
            for it in range(12):
                if used >= budget_left or time.time() > deadline:
                    break
                step = (0.05 + 0.05 * np.random.rand()) * (hi - lo)
                y = best + step * np.random.standard_normal(dim)
                y = np.clip(y, lo, hi)
                fy = np.array(problem(y))
                used += 1
                if not np.all(fy < ref):
                    continue
                # accept if not dominated by bestf
                if not (np.all(bestf <= fy) and np.any(bestf < fy)):
                    best = y
                    bestf = fy
                    fx = fy
            x = best
            fx = bestf
            if used >= budget_left or time.time() > deadline:
                break
        return x, used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin well under 330s

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        # reserve budget for polish
        polish_reserve = 40
        n_polish = min(polish_reserve, max(0, budget - 10))
        n_evo = max(1, budget - n_polish)

        # ---- initial population: LHS + random ----
        n_lhs = 30
        n_rand = 10
        n_init = min(n_lhs + n_rand, n_evo)
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

        # archive = non-dominated among valid points
        valid = np.all(F_init < ref, axis=1)
        pF = F_init[valid]
        pX = X_init[valid]
        if len(pF) > 0:
            pmask = self._is_pareto(pF)
            archive_X = pX[pmask].copy()
            archive_F = pF[pmask].copy()
        else:
            archive_X = np.empty((0, dim))
            archive_F = np.empty((0, n_obj))

        # population for evolution (use valid points + a few random if needed)
        popX = X_init.copy()
        popF = F_init.copy()

        # ---- evolution loop (fast non-dominated sort + crowding) ----
        pop_size = len(popX)
        while used < n_evo:
            if time.time() > deadline - 5:
                break
            # select parents (elite: top of archive + random)
            children = []
            for _ in range(2):
                if len(archive_F) > 0:
                    if np.random.rand() < 0.7:
                        # pick from archive weighted by crowding
                        cd = self._crowding_distance(archive_F)
                        cd = np.where(np.isinf(cd), 0, cd)
                        w = cd + 1e-9
                        w = w / w.sum()
                        pi = np.random.choice(len(archive_F), p=w)
                        parent = archive_X[pi]
                    else:
                        parent = archive_X[np.random.randint(len(archive_F))]
                elif len(popX) > 0:
                    parent = popX[np.random.randint(len(popX))]
                else:
                    parent = lo + np.random.uniform(size=dim) * (hi - lo)
                # mutation
                progress = used / max(1, n_evo)
                base = 0.6 * (1.0 - progress * 0.6) + 0.1
                step = base * (0.5 + np.random.uniform())
                if np.random.uniform() < 0.4:
                    child = parent + step * (hi - lo) * np.random.standard_normal(dim)
                else:
                    child = parent.copy()
                    nc = np.random.randint(1, 3)
                    coords = np.random.choice(dim, size=nc, replace=False)
                    for c in coords:
                        child[c] = parent[c] + step * (hi[c] - lo[c]) * np.random.standard_normal()
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                children.append((child, fc))
                if used >= n_evo:
                    break

            # add children to population
            for (child, fc) in children:
                popX = np.vstack([popX, [child]])
                popF = np.vstack([popF, [fc]])
                # update archive
                if np.all(fc < ref):
                    if not self._dominates(fc, archive_F).any() if len(archive_F) else True:
                        dominated = len(archive_F) > 0 and np.all(
                            np.any(archive_F <= fc, axis=1) &
                            np.any(archive_F < fc, axis=1))
                        if not dominated:
                            archive_X = np.vstack([archive_X, [child]])
                            archive_F = np.vstack([archive_F, [fc]])
                            keep = np.ones(len(archive_F), dtype=bool)
                            for i in range(len(archive_F) - 1):
                                if np.all(archive_F[i] >= fc) and np.any(archive_F[i] > fc):
                                    keep[i] = False
                            archive_X = archive_X[keep]
                            archive_F = archive_F[keep]

            # rebuild population via fast non-dominated sort + crowding
            if len(popF) > 0:
                # keep only valid for selection
                vmask = np.all(popF < ref, axis=1)
                vF = popF[vmask]
                vX = popX[vmask]
                if len(vF) > 0:
                    fronts = self._fast_nondominated_sort(vF)
                    new_idx = []
                    for front in fronts:
                        if len(new_idx) + len(front) <= pop_size:
                            new_idx.extend(front)
                        else:
                            cd = self._crowding_distance(vF[front])
                            order = np.argsort(-cd)
                            take = pop_size - len(new_idx)
                            new_idx.extend([front[i] for i in order[:max(0, take)]])
                            break
                    if new_idx:
                        popF = vF[new_idx]
                        popX = vX[new_idx]

        # ---- polish phase: local search on top archive points ----
        if used < budget and len(archive_F) > 0:
            # rank archive by crowding (spread) and take top few
            cd = self._crowding_distance(archive_F)
            cd = np.where(np.isinf(cd), 0, cd)
            order = np.argsort(-cd)
            top = order[:min(8, len(order))]
            for pi in top:
                if used >= budget or time.time() > deadline - 2:
                    break
                parent = archive_X[pi]
                newx, nused = self._local_search(parent, problem, lo, hi, ref,
                                                 budget - used, deadline)
                used += nused
                if nused > 0:
                    nf = np.array(problem(newx))
                    if np.all(nf < ref):
                        dominated = len(archive_F) > 0 and np.all(
                            np.any(archive_F <= nf, axis=1) &
                            np.any(archive_F < nf, axis=1))
                        if not dominated:
                            archive_X = np.vstack([archive_X, [newx]])
                            archive_F = np.vstack([archive_F, [nf]])
                            keep = np.ones(len(archive_F), dtype=bool)
                            for i in range(len(archive_F) - 1):
                                if np.all(archive_F[i] >= nf) and np.any(archive_F[i] > nf):
                                    keep[i] = False
                            archive_X = archive_X[keep]
                            archive_F = archive_F[keep]

        # record all evaluated points (return value ignored, but keep consistent)
        return popX

# EVOLVE-BLOCK-END
