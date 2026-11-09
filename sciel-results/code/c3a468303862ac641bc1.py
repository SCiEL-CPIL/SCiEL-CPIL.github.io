"""Fresh multi-objective optimizer for RE34 (5 vars, 3 objectives, box [1,3])."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DE + non-dominated archive with crowding-distance replacement and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 315.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        rng = np.random.default_rng(0)

        # ---------- non-dominated archive ----------
        archive = []  # list of (x, f)

        def _is_dom(f):
            # True if any archived point dominates f
            if not archive:
                return False
            F = np.array([a[1] for a in archive])
            return bool(np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)))

        def _add(x, f):
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return False
            if _is_dom(f):
                return False
            # remove archive points dominated by f
            new_arch = []
            for (ax, af) in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_arch.append((ax, af))
            new_arch.append((x, f))
            archive[:] = new_arch
            return True

        def _crowding():
            n = len(archive)
            if n <= 2:
                return np.full(n, np.inf)
            F = np.array([a[1] for a in archive])
            cd = np.zeros(n)
            for k in range(n_obj):
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rngv = F[order[-1], k] - F[order[0], k]
                if rngv > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / rngv
            return cd

        def _trim(maxn):
            if len(archive) <= maxn:
                return
            cd = _crowding()
            keep = len(archive) - maxn
            # drop the `keep` points with the smallest crowding distance
            drop = np.argsort(cd)[:keep]
            dropset = set(int(i) for i in drop)
            archive[:] = [a for i, a in enumerate(archive) if i not in dropset]

        used = 0

        def _eval(x):
            nonlocal used
            f = problem(np.asarray(x, dtype=float))
            used += 1
            return f

        MAXARCH = 220

        # ---------- 1. Latin-hypercube seed (diverse) ----------
        def _lhs(n):
            rows = np.arange(n)
            P = []
            for j in range(dim):
                perm = rng.permutation(n)
                P.append((perm + rng.random(n)) / n)
            return lo + span * np.array(P).T  # (n, dim)

        n_seed = min(budget, 30)
        Xs = _lhs(n_seed)
        for x in Xs:
            if used >= budget:
                break
            f = _eval(x)
            _add(x.copy(), f)
        _trim(MAXARCH)

        # ---------- 2. cheap corner sampling (objective extremes) ----------
        if used < budget:
            corners = []
            for c in ((0, 0), (0, 1), (1, 0), (1, 1)):
                cc = np.array([lo[j] if c[j] == 0 else hi[j] for j in range(dim)])
                corners.append(cc)
            for c in corners:
                if used >= budget:
                    break
                f = _eval(c)
                _add(c.copy(), f)
            _trim(MAXARCH)

        # ---------- 3. Differential Evolution ----------
        def _de_step(step):
            nonlocal used
            if len(archive) < 3:
                return
            if used >= budget:
                return
            if time.time() - t0 > deadline:
                return
            # pick 3 distinct parents
            idx = rng.choice(len(archive), size=3, replace=False)
            a = archive[int(idx[0])][0]
            b = archive[int(idx[1])][0]
            c = archive[int(idx[2])][0]
            v = a + step * (b - c)
            v = np.clip(v, lo, hi)
            f = _eval(v)
            if _add(v.copy(), f):
                _trim(MAXARCH)

        n_iter = budget - used
        step = 0.9
        decay = 0.997
        for it in range(n_iter):
            if used >= budget:
                break
            if time.time() - t0 > deadline:
                break
            _de_step(step)
            step *= decay

        # ---------- 4. spare-time LHS sweep + local polish ----------
        def _local_polish(x, f, n_try=6, step=0.05):
            nonlocal used
            for _ in range(n_try):
                if used >= budget:
                    break
                if time.time() - t0 > deadline:
                    break
                d = rng.standard_normal(dim)
                nx = np.clip(x + step * span * d, lo, hi)
                nf = _eval(nx)
                # keep if not dominated (archive handles dominance)
                _add(nx.copy(), nf)
            _trim(MAXARCH)

        # gap-filling LHS in remaining budget
        if used < budget and time.time() - t0 < deadline:
            n_l = min(budget - used, 40)
            if n_l > 0:
                Xl = _lhs(n_l)
                for x in Xl:
                    if used >= budget:
                        break
                    if time.time() - t0 > deadline:
                        break
                    f = _eval(x)
                    _add(x.copy(), f)
                _trim(MAXARCH)

        # polish the current front (improve / densify around known good points)
        if used < budget and time.time() - t0 < deadline:
            for (x, f) in list(archive):
                if used >= budget:
                    break
                if time.time() - t0 > deadline:
                    break
                _local_polish(x, f, n_try=4, step=0.04)

        return archive
# EVOLVE-BLOCK-END
