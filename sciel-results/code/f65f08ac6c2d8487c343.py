"""Model-and-optimize optimizer for BBOB f19 (Griewank-Rosenbrock).

A Gaussian-process surrogate (model) is fitted to observed (x, f) pairs.  The
next query is chosen by optimizing an acquisition function (probability-of-
improvement / expected-improvement / upper-confidence-bound) over the search
box (optimize).  A proper CMA-ES (with covariance adaptation) is run as a
multi-start final local refinement, which is what captures the rotated
landscape structure that a diagonal (1+1)-ES misses.  A hard wall-clock
deadline guard returns the best point found within the budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import erf


def _N_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _N_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


class Optimizer:
    """GP surrogate (model) + multi-acquisition optimize + CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- sampling / init ------------------------------------------------
    def _init_point(self, rng):
        x = np.zeros(self.dim)
        r = rng.random()
        if r < 0.2:
            x = rng.uniform(-5, 5, self.dim)
        elif r < 0.4:
            x = rng.integers(-3, 4, self.dim).astype(float)
        else:
            x = (rng.random(self.dim) < 0.5) * 0.0
        return x

    def _candidates(self, rng, n, mu, spread):
        cands = [np.clip(mu + spread * rng.standard_normal(self.dim), -5, 5)
                 for _ in range(n)]
        cands.append(rng.uniform(-5, 5, self.dim))
        cands.append(mu.copy())
        return cands

    # ---- GP model -------------------------------------------------------
    def _gp_fit(self, X, y):
        n = X.shape[0]
        if n == 1:
            return float(y[0]), np.zeros(1), 1.0
        d = X - X[0]
        d2 = (d * d).sum(axis=1)
        nonzero = d2[d2 > 0]
        if nonzero.size:
            l = float(np.clip(np.sqrt(2.0 * np.mean(nonzero)), 0.05, 10.0))
        else:
            l = 1.0
        l2 = l * l
        K = np.exp(-0.5 * d2 / l2)
        K[np.arange(n), np.arange(n)] += 1e-6
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-4 * np.eye(n))
        a = np.linalg.solve(L, np.ones(n))
        y0 = float(np.mean(y))
        return y0, a, l2

    def _gp_predict(self, X, Xq, y0, a, l2):
        n = X.shape[0]
        dx = Xq[:, None, :] - X[None, :, :]
        d2 = (dx * dx).sum(axis=2)
        Kq = np.exp(-0.5 * d2 / l2)                      # (q, n)
        mu = y0 + Kq @ a
        v = Kq * a[None, :]
        var = float(a @ a) - (v * v).sum(axis=1)
        return mu, np.maximum(var, 0.0)

    def _grad(self, X, x, y0, a, l2):
        # gradient of (y0 + Kq(x)@a) w.r.t. x = sum_j a_j * Kq_j * (x - X_j)/l2
        dx = x[None, :] - X                                # (n, d)
        d2 = (dx * dx).sum(axis=1)
        Kq = np.exp(-0.5 * d2 / l2)                        # (n,)
        g = (a[None, :] * Kq[:, None] * dx / l2).sum(axis=0)
        return g

    # ---- acquisition (value + gradient) ---------------------------------
    def _acq(self, X, Xq, b, strategy, gamma, y0, a, l2):
        mu, var = self._gp_predict(X, Xq, y0, a, l2)
        sd = np.sqrt(np.maximum(var, 1e-12))
        z = (b - mu) / np.where(sd < 1e-9, 1e-9, sd)
        if strategy == 0:      # probability of improvement
            val = _N_cdf(z)
        elif strategy == 1:    # expected improvement
            phi = _N_pdf(z)
            Phi = _N_cdf(z)
            ei = (b - mu) * (1.0 - Phi) + sd * phi
            val = np.maximum(ei, 0.0)
        else:                  # upper confidence bound (minimize)
            val = mu - gamma * sd
        return val

    def _acq_grad(self, X, b, strategy, gamma, y0, a, l2, x):
        mu, var = self._gp_predict(X, x[None, :], y0, a, l2)
        mu = float(mu[0])
        sd = float(np.sqrt(max(var[0], 1e-12)))
        g = self._grad(X, x, y0, a, l2)
        Kq = np.exp(-0.5 * ((x[None, :] - X) ** 2).sum(axis=1) / l2)
        if sd < 1e-9:
            if strategy == 0:
                return -mu, -g
            if strategy == 1:
                return -mu, -g
            return -mu, -g
        z = (b - mu) / sd
        if strategy == 0:
            phi = _N_pdf(z)
            val = _N_cdf(z)
            gv = phi / sd
            gacq = gv * g
        elif strategy == 1:
            phi = _N_pdf(z)
            Phi = _N_cdf(z)
            c = max((b - mu) * (1.0 - Phi) + sd * phi, 0.0)
            val = c
            dmu = -(1.0 - Phi)
            dsd = phi
            vK = Kq * a
            dvar = -2.0 * (vK * Kq) @ ((x[None, :] - X) / l2)
            dsdx = 0.5 / sd * dvar
            gacq = dmu * g + dsdx * g
        else:
            val = mu - gamma * sd
            vK = Kq * a
            dvar = -2.0 * (vK * Kq) @ ((x[None, :] - X) / l2)
            dsdx = 0.5 / sd * dvar
            gacq = g - gamma * dsdx
        return -val, -gacq

    def _acq_starts(self, rng, X, b, nstarts):
        starts = []
        # incumbent (best point)
        bi = int(np.argmin(np.array([self._f_hist[i] for i in range(len(X))])
                           if False else np.array(self._ys)))
        starts.append(np.clip(X[bi], -5, 5))
        # a few random incumbent neighbourhoods
        for _ in range(max(1, nstarts // 4)):
            starts.append(np.clip(X[bi] + 0.6 * rng.standard_normal(self.dim), -5, 5))
        # Latin-hypercube spread over the box
        n = self.dim
        for _ in range(nstarts - len(starts)):
            u = rng.random(n)
            x = -5 + 10 * (u + rng.random(n)) / 2.0
            starts.append(np.clip(x, -5, 5))
        return starts[:nstarts]

    def _optimize_acq(self, X, b, strategy, gamma, y0, a, l2, rng, nstarts, maxiter):
        from scipy.optimize import minimize as _min
        best_val = -np.inf
        best_x = None
        starts = self._acq_starts(rng, X, b, nstarts)
        for x0 in starts:
            x0 = np.clip(x0, -5, 5)
            try:
                res = _min(self._acq_grad, x0,
                           args=(X, b, strategy, gamma, y0, a, l2),
                           method="L-BFGS-B", bounds=[(-5, 5)] * self.dim,
                           jac=True, options={"maxiter": maxiter, "gtol": 1e-4})
                xv = np.clip(res.x, -5, 5)
                val, _ = self._acq_grad(X, b, strategy, gamma, y0, a, l2, xv)
                if val < best_val:
                    best_val = val
                    best_x = xv
            except Exception:
                continue
        if best_x is None:
            best_x = starts[0]
        return best_x

    # ---- proper CMA-ES (with covariance adaptation) ---------------------
    def _cmaes(self, x0, f0, rng, budget, deadline):
        n = self.dim
        pop = max(4, int(4 + 3 * np.log(n)))
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        c = np.array(x0, dtype=float)
        sigma = 0.35
        C = np.eye(n)
        Cinv = np.eye(n)
        Csqrt = np.eye(n)
        Cinv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        best_x, best_f = c.copy(), f0
        it = 0
        while it < budget and time.time() < deadline:
            Y = np.zeros((pop, n))
            F = np.empty(pop)
            for i in range(pop):
                z = rng.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                # margin-clipping to keep points strictly inside the box
                lo = -5 + 0.05
                hi = 5 - 0.05
                x = np.clip(x, lo, hi)
                F[i] = float(self._f(x))
                Y[i] = y
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]
            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean
            c = c + cs * (d / cs - c)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
            hsig = (np.linalg.norm(p_c) /
                    np.sqrt(1 - (1 - cs) ** (it + 1)) <
                    (1.4 + 2 / (n + 1)) * chiN)
            hsig = float(hsig)
            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 50.0))
            # refresh square roots
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
                Cinv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
            except np.linalg.LinAlgError:
                Csqrt = np.eye(n)
                Cinv_sqrt = np.eye(n)
            c = np.clip(c, -5 + 0.05, 5 - 0.05)
            it += 1
        return best_x, best_f

    # ---- main -----------------------------------------------------------
    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 250.0          # safety margin well under 300/330s
        rng = np.random.default_rng(0)
        self._f = func
        self._ys = []

        x0 = self._init_point(rng)
        xs = [x0.copy()]
        ys = [float(func(x0))]
        self._ys = list(ys)
        best = (xs[0].copy(), float(ys[0]))
        calls = 1

        def sample(x):
            nonlocal best, calls
            f = float(func(x))
            xs.append(np.asarray(x, dtype=float).copy())
            ys.append(f)
            self._ys.append(f)
            calls += 1
            if f < best[1]:
                best = (xs[-1].copy(), f)
            return f

        # Phase 1: short (1+1)-ES to seed a local, informative set.
        x = x0.copy()
        fx = ys[0]
        sigma = 2.5
        n1 = max(8, int(0.05 * self.budget))
        for _ in range(n1):
            if time.time() >= deadline or calls >= self.budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = sample(y)
            if fy < fx:
                x, fx = y.copy(), fy
                sigma *= 1.25
            else:
                sigma *= 0.85

        # Phase 2: model-and-optimize loop with mixed acquisition strategies.
        strat_cycle = [1, 0, 2, 1, 2, 0]   # EI, PI, UCB, ...
        si = 0
        while calls < self.budget and time.time() < deadline:
            n = len(xs)
            if n >= 4:
                X = np.array(xs)
                y = np.array(ys, dtype=float)
                y0, a, l2 = self._gp_fit(X, y)
                b = float(best[1])
                strategy = strat_cycle[si % len(strat_cycle)]
                si += 1
                gamma = 1.5
                remain = self.budget - calls
                nstarts = 8 if n < 60 else 5
                maxiter = 30
                xq = self._optimize_acq(X, b, strategy, gamma, y0, a, l2,
                                        rng, nstarts, maxiter)
            else:
                xq = np.clip(best[0] + 1.5 * rng.standard_normal(self.dim), -5, 5)
            if time.time() >= deadline or calls >= self.budget:
                break
            sample(xq)

        # Phase 3: multi-start local CMA-ES polish around the best points.
        if calls < self.budget and time.time() < deadline - 40:
            # pick up to 3 distinct good starting points
            order = np.argsort(ys)
            seen = []
            starts = []
            for i in order:
                xi = xs[i]
                if all(np.linalg.norm(xi - s) > 0.3 for s in seen):
                    seen.append(xi)
                    starts.append((xi.copy(), float(ys[i])))
                if len(starts) >= 3:
                    break
            if not starts:
                starts = [(best[0].copy(), best[1])]
            per = max(1, (self.budget - calls) // len(starts))
            for sx, sf in starts:
                if calls >= self.budget or time.time() >= deadline:
                    break
                px, pf = self._cmaes(sx, sf, rng, per, deadline)
                if pf < best[1]:
                    best = (px.copy(), pf)

        return best
# EVOLVE-BLOCK-END
