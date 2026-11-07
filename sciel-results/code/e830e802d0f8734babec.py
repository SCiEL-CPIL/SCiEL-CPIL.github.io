"""CMA-ES + structure-exploiting Hessian-fit optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a quadratic-structure (Hessian-fit) refinement for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---------- CMA-ES building blocks ----------
    def _chol_update(self, C):
        """Cholesky factorization with fallback to identity."""
        n = C.shape[0]
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = C + 1e-12 * np.eye(n)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
        B = L.T                      # search directions (columns)
        D = np.sqrt(np.diag(C))
        Dinv = 1.0 / D
        return B, D, Dinv

    def _cmaes_single(self, func, rng, x0, sigma0, max_calls, deadline,
                      global_best_x, global_best_f, calls):
        """
        Run a single CMA-ES instance.
        Returns (best_x, best_f).  'calls' is a shared counter (list).
        """
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        cc = (4 + mueff / n) / ((n + 4) + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc + cs)
        cmu = min(2 * (mueff - 1) * (0.5 + cs / cc) / ((n + 2) ** 2 + mueff),
                  c1 * (3 - cc))

        xmean = np.array(x0, dtype=float)
        sigma = float(sigma0)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B, D, Dinv = self._chol_update(C)
        chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        best_x = np.array(global_best_x, dtype=float)
        best_f = float(global_best_f)
        stagnation = 0
        best_hist = best_f

        while calls[0] < max_calls:
            if time.time() > deadline:
                break

            z = rng.standard_normal((lam, n))
            pops = xmean[None, :] + sigma * (z @ B.T)
            pops = np.clip(pops, -5, 5)

            vals = np.full(lam, np.inf)
            for i in range(lam):
                if time.time() > deadline or calls[0] >= max_calls:
                    break
                calls[0] += 1
                try:
                    vals[i] = float(func(pops[i]))
                except Exception:
                    vals[i] = np.inf

            if calls[0] >= max_calls:
                break

            order = np.argsort(vals)
            for i in order:
                if vals[i] < best_f:
                    best_f = vals[i]
                    best_x = pops[i].copy()

            idx = order[:mu]
            y = (pops[idx] - xmean[None, :]) / sigma
            yw = (w[:, None] * y).sum(axis=0)
            xnew = xmean + sigma * yw

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Dinv @ (B.T @ yw))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (calls[0] + 1) / (n + cs)))) < \
                   ((1 + 2 / (n + 1)) * chiN)
            pc = (1 - cc) * pc + hsig * cc * yw

            rank1 = np.outer(yw, yw)
            rankmu = (w[:, None, None] * np.einsum('ij,ik->ijk', y, y)).sum(axis=0)
            C = ((1 + c1 * cc * (cc - 1) - c1 - cmu) * C +
                 c1 * (rank1 + (cc - cc * cc) * C) +
                 cmu * rankmu)
            C = 0.5 * (C + C.T)
            B, D, Dinv = self._chol_update(C)
            xmean = xnew

            sigma = sigma * np.exp(cs / (5 * n) * (norm_ps / chiN - 1))
            sigma = float(np.clip(sigma, 1e-12, 1e3))

            if norm_ps < 1e-10 and sigma < 1e-8:
                break

            if best_f < best_hist - 1e-10:
                best_hist = best_f
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > 2 * lam:
                break

        return best_x, best_f

    # ---------- structure-exploiting refinement ----------
    def _fit_and_minimize(self, func, rng, x, fx, budget_left, deadline, calls):
        """
        Fit the quadratic (Hessian) at x from a small directional sample,
        then 1-D minimize along each recovered direction.
        Returns (best_x, best_f).
        """
        n = self.dim
        best_x = np.array(x, dtype=float)
        best_f = float(fx)

        # directional sample: 2*n points (each axis +/- a) + origin
        a = 0.35
        dirs = np.zeros((2 * n, n))
        for i in range(n):
            dirs[2 * i, i] = a
            dirs[2 * i + 1, i] = -a
        fv = np.empty(2 * n + 1)
        fv[2 * n] = best_f  # f(x)
        for i in range(2 * n):
            if time.time() > deadline or calls[0] >= budget_left:
                fv[i] = np.inf
                continue
            calls[0] += 1
            pt = np.clip(best_x + dirs[i], -5, 5)
            try:
                fv[i] = float(func(pt))
            except Exception:
                fv[i] = np.inf

        # least-squares fit of symmetric quadratic
        A = np.zeros((2 * n, n * (n + 1) // 2 + n + 1))
        for i in range(2 * n):
            d = dirs[i]
            j = 0
            for r in range(n):
                for c in range(r, n):
                    val = d[r] * d[c]
                    if r != c:
                        val *= 2.0
                    A[i, j] = val
                    j += 1
            A[i, n * (n + 1) // 2:n * (n + 1) // 2 + n] = d
            A[i, -1] = 1.0
        bvec = fv[:2 * n] - fv[2 * n]

        try:
            coef, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        except np.linalg.LinAlgError:
            coef = None

        if coef is not None:
            H = np.zeros((n, n))
            j = 0
            for r in range(n):
                for c in range(r, n):
                    val = coef[j]
                    H[r, c] = val
                    if r != c:
                        H[c, r] = val
                    j += 1
            g = coef[n * (n + 1) // 2:n * (n + 1) // 2 + n]

            # eigendecompose the Hessian -> search directions and scales
            try:
                w, V = np.linalg.eigh(0.5 * (H + H.T))
            except np.linalg.LinAlgError:
                w, V = None, None
            if w is not None:
                for k in range(n):
                    lam_k = w[k]
                    if lam_k < 1e-12:
                        continue
                    d = V[:, k]
                    # step scale: along direction d the 1-D quadratic min is
                    # at t* = -g.d / lam_k ; choose a bracket half that scale
                    tstar = -float(np.dot(g, d)) / lam_k
                    t0 = min(3.0, max(0.05, 0.5 * abs(tstar)))
                    if t0 < 1e-4:
                        continue
                    # golden-section 1-D minimization over t in [-t0, t0]
                    lo, hi = -t0, t0
                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                    c1p = hi - gr * (hi - lo)
                    c2p = lo + gr * (hi - lo)
                    pt1 = np.clip(best_x + c1p * d, -5, 5)
                    pt2 = np.clip(best_x + c2p * d, -5, 5)
                    calls[0] += 1
                    f1 = float(func(pt1))
                    if time.time() > deadline or calls[0] >= budget_left:
                        break
                    calls[0] += 1
                    f2 = float(func(pt2))
                    for _ in range(12):
                        if time.time() > deadline or calls[0] >= budget_left:
                            break
                        if f1 < f2:
                            hi = c2p
                            c2p = c1p
                            f2 = f1
                            c1p = hi - gr * (hi - lo)
                            pt1 = np.clip(best_x + c1p * d, -5, 5)
                            calls[0] += 1
                            f1 = float(func(pt1))
                        else:
                            lo = c1p
                            c1p = c2p
                            f1 = f2
                            c2p = lo + gr * (hi - lo)
                            pt2 = np.clip(best_x + c2p * d, -5, 5)
                            calls[0] += 1
                            f2 = float(func(pt2))
                    # take the better of the two bracket points
                    tc = 0.5 * (lo + hi)
                    ptc = np.clip(best_x + tc * d, -5, 5)
                    for t, ft in ((c1p, f1), (c2p, f2), (tc, None)):
                        if ft is None:
                            if time.time() > deadline or calls[0] >= budget_left:
                                break
                            calls[0] += 1
                            ft = float(func(ptc))
                        if ft < best_f:
                            best_f = ft
                            best_x = np.clip(best_x + t * d, -5, 5).copy()

        # final jitter polish
        jitter = 1e-3
        for _ in range(20):
            if time.time() > deadline or calls[0] >= budget_left:
                break
            cand = np.clip(best_x + jitter * rng.standard_normal(n), -5, 5)
            calls[0] += 1
            try:
                cf = float(func(cand))
            except Exception:
                cf = np.inf
            if cf < best_f:
                best_f = cf
                best_x = cand.copy()
                jitter *= 0.5
            else:
                jitter *= 0.7
            if jitter < 1e-8:
                break

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        total_budget = self.budget
        n = self.dim
        rng = np.random.default_rng(42)
        calls = [0]

        best_x = None
        best_f = np.inf

        def evaluate(x):
            calls[0] += 1
            try:
                return float(func(np.clip(x, -5, 5)))
            except Exception:
                return np.inf

        # ---- Phase 0: cheap diverse starts to seed the best point ----
        starts = [np.zeros(n)]
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, n))
        for s in starts:
            if calls[0] >= total_budget or time.time() > deadline:
                break
            fs = evaluate(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        # ---- Phase 1: CMA-ES with adaptive restarts (exploits anisotropy) ----
        num_starts = 4
        cma_starts = [best_x.copy()]
        for _ in range(num_starts - 1):
            cma_starts.append(rng.uniform(-4, 4, n))
        b1 = int(total_budget * 0.35)
        rem = (total_budget - b1) // (num_starts - 1)
        budgets = [b1] + [rem] * (num_starts - 1)
        budgets[-1] += total_budget - sum(budgets)

        for si, (x0, b) in enumerate(zip(cma_starts, budgets)):
            if calls[0] >= total_budget or time.time() > deadline:
                break
            sigma0 = 2.0 if si == 0 else 1.5
            bx, bf = self._cmaes_single(
                func, rng, x0, sigma0, total_budget, deadline,
                best_x, best_f, calls
            )
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            # structure-exploiting refinement after each CMA-ES run
            if calls[0] < total_budget and time.time() < deadline:
                bx, bf = self._fit_and_minimize(
                    func, rng, best_x, best_f, total_budget, deadline, calls
                )
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        # ---- Phase 2: repeated Hessian-fit refinement (different modelling family) ----
        for _ in range(3):
            if calls[0] >= total_budget or time.time() > deadline:
                break
            bx, bf = self._fit_and_minimize(
                func, rng, best_x, best_f, total_budget, deadline, calls
            )
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
