# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Restartable CMA-ES with Nelder-Mead / coordinate / golden-section polish
    for BBOB f9 (rotated Rosenbrock) on [-5,5]^5.

    Strategy:
      0. Evaluate a few cheap seeds to seed a good incumbent + CMA center.
      1. Run restartable CMA-ES to find the basin (robust global search).
      2. Polish the best point with Nelder-Mead, coordinate descent and
         golden-section line search.
    Explicit wall-clock deadline guard; the return value is ignored by the
    evaluator, so we always return the best incumbent found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t_start = time.time()
        deadline = t_start + 300.0  # evaluator kills at ~330s; keep margin

        # ---- state ----
        best_x = None
        best_f = np.inf
        calls_used = 0

        def timed():
            return (time.time() - t_start) > (deadline - t_start)

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        def evalx(x):
            nonlocal best_x, best_f, calls_used
            x = np.asarray(x, dtype=float)
            x = clip_box(x)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ================= Phase 0: cheap seeds =================
        seeds = [
            np.ones(dim),
            np.zeros(dim),
            np.full(dim, 1.0),
            -np.ones(dim),
            np.linspace(-4, 4, dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
        ]
        rng = np.random.default_rng(12345)
        for _ in range(8):
            seeds.append(rng.uniform(-5, 5, dim))
        for s in seeds:
            if calls_used >= budget or timed():
                break
            evalx(s)

        center = best_x.copy() if best_x is not None else np.zeros(dim)

        # ================= Phase 1: restartable CMA-ES =================
        # CMA-ES parameters (dim=5 -> population ~ 4 + 3*ln(5) ~ 9)
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = max(1, lam // 2)
        # recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * min(1, mu_eff / dim) - c_c) / (dim + 1 + 2 * mu_eff / dim)

        def run_cma(center, sigma0, max_calls):
            nonlocal calls_used
            x = np.asarray(center, dtype=float).copy()
            sigma = float(sigma0)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # hessian of identity (delta_ik * (2 c_c / c_cov - 1 + c_cov))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            it = 0
            while calls_used < budget and calls_used < max_calls and not timed():
                it += 1
                # sample population
                Y = rng.standard_normal((lam, dim))
                pops = np.array([clip_box(x + sigma * (B @ (D * y).T).T) for y in Y])
                fvals = np.empty(lam)
                for i in range(lam):
                    if calls_used >= budget or timed():
                        break
                    fvals[i] = func(pops[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pops[i].copy()
                if calls_used >= budget or timed():
                    break
                # select best mu
                order = np.argsort(fvals)
                sel = order[:mu]
                m = np.average(pops[sel], axis=0, weights=w)
                # step-size adaptation
                yvec = (m - x) / (sigma * np.sqrt(np.sum((D * B.T @ p_sigma) ** 2) if False else (B @ (D * p_sigma)) ** 2))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yvec
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                sigma = min(max(sigma, 1e-12), 10.0)
                # update mean
                x = m
                # path covariance
                y_mean = (m - x) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                C = B @ (D ** 2) @ B.T
                hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
                C = (1 + c_cov * (2 * hsig - 1) - c_cov) * C \
                    + c_cov * (np.outer(p_c, p_c) + (hsig - 1) * (1 - c_c) * C)
                # ensure SPD and recompute eigen
                C = 0.5 * (C + C.T)
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.clip(D2, 1e-20, None))
                except np.linalg.LinAlgError:
                    D2, B2 = np.ones(dim), np.eye(dim)
                B, D = B2, D2
                # stagnation restart: no improvement for many iterations -> break
                if it % 40 == 0:
                    # if sigma collapsed or mean stuck, break to restart
                    if sigma < 1e-6:
                        break
            return x, sigma

        # CMA-ES with restarts from best point
        cma_restarts = 6
        cma_sigma = 2.0
        for r in range(cma_restarts):
            if calls_used >= budget or timed():
                break
            # center: first restart from best, later restarts from random perturbation
            if r == 0:
                center = best_x.copy() if best_x is not None else np.zeros(dim)
                sigma0 = cma_sigma
            else:
                center = best_x + rng.standard_normal(dim) * cma_sigma
                center = clip_box(center)
                sigma0 = cma_sigma * (1.0 + 0.3 * r)
            max_calls = min(budget - calls_used, 250)
            run_cma(center, sigma0, max_calls)
            cma_sigma = min(cma_sigma * 0.7, 0.2)

        # ================= Phase 2: local polish =================
        if best_x is not None and calls_used < budget and not timed():
            # --- Nelder-Mead ---
            def nelder_mead(x0, max_calls):
                nonlocal calls_used
                x = np.asarray(x0, dtype=float).copy()
                f0 = func(x); calls_used += 1
                if f0 < best_f:
                    best_f = f0; best_x = x.copy()
                # simplex
                simplex = [x.copy()]
                fvals = [f0]
                for d in range(dim):
                    xp = x.copy()
                    step = 0.5 if xp[d] < 0 else -0.5
                    xp[d] += step
                    xp = clip_box(xp)
                    fp = func(xp); calls_used += 1
                    if fp < best_f:
                        best_f = fp; best_x = xp.copy()
                    simplex.append(xp); fvals.append(fp)
                    if calls_used >= max_calls or timed():
                        return
                alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
                it = 0
                while calls_used < max_calls and not timed() and it < 200:
                    it += 1
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    xw = simplex[-1]; fw = fvals[-1]
                    xbest = simplex[0]; fbest = fvals[0]
                    centroid = np.mean(simplex[:-1], axis=0)
                    # reflect
                    xr = clip_box(centroid + alpha * (centroid - xw))
                    fr = func(xr); calls_used += 1
                    if fr < fbest:
                        # expand
                        xe = clip_box(centroid + gamma * (centroid - xw))
                        fe = func(xe); calls_used += 1
                        if fe < fr:
                            simplex[-1], fvals[-1] = xe, fe
                        else:
                            simplex[-1], fvals[-1] = xr, fr
                    elif fr < fvals[-2]:
                        simplex[-1], fvals[-1] = xr, fr
                    else:
                        # contract
                        if fr < fw:
                            xc = clip_box(centroid + 0.5 * (xr - centroid))
                            fc = func(xc); calls_used += 1
                            if fc <= fr:
                                simplex[-1], fvals[-1] = xc, fc
                            else:
                                # shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = clip_box(xbest + sigma_nm * (simplex[i] - xbest))
                                    fi = func(simplex[i]); calls_used += 1
                                    fvals[i] = fi
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = clip_box(xbest + sigma_nm * (simplex[i] - xbest))
                                fi = func(simplex[i]); calls_used += 1
                                fvals[i] = fi
                    if calls_used >= max_calls or timed():
                        break
                    # update best
                    for i in range(len(simplex)):
                        if fvals[i] < best_f:
                            best_f = fvals[i]; best_x = simplex[i].copy()

            nelder_mead(best_x, min(budget - calls_used, 120))

            # --- coordinate descent with golden-section line search ---
            def golden(a, b, fval, x, d):
                nonlocal calls_used
                gr = (np.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d_ = a + gr * (b - a)
                fc = fval(c); fd = fval(d_)
                for _ in range(8):
                    if calls_used >= budget or timed():
                        break
                    if fc < fd:
                        b, d_ = d_, c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = fval(c)
                    else:
                        a, c = c, d_
                        fc = fd
                        d_ = a + gr * (b - a)
                        fd = fval(d_)
                return (a + b) / 2

            if best_x is not None and calls_used < budget and not timed():
                x = best_x.copy()
                fx = func(x); calls_used += 1
                if fx < best_f:
                    best_f = fx; best_x = x.copy()
                for _ in range(12):
                    if calls_used >= budget or timed():
                        break
                    improved = False
                    for d in range(dim):
                        if calls_used >= budget or timed():
                            break
                        # find a step direction with improvement via small search
                        best_step = 0.0
                        base = fx
                        for sgn in (1, -1):
                            step = 0.5
                            for _t in range(6):
                                if calls_used >= budget or timed():
                                    break
                                cand = x.copy(); cand[d] += sgn * step
                                cand = clip_box(cand)
                                fc = func(cand); calls_used += 1
                                if fc < base:
                                    base = fc
                                    best_step = sgn * step
                                    # golden refine around this
                                    lo = x[d] - sgn * step * 2 if sgn > 0 else x[d] + sgn * step * 2
                                    hi = x[d] + sgn * step * 2 if sgn > 0 else x[d] - sgn * step * 2
                                    lo, hi = sorted([lo, hi])
                                    def linef(t):
                                        zz = x.copy(); zz[d] = t
                                        zz = clip_box(zz)
                                        return func(zz)
                                    ts = golden(lo, hi, linef, x, d)
                                    cand2 = x.copy(); cand2[d] = ts
                                    cand2 = clip_box(cand2)
                                    fc2 = func(cand2); calls_used += 1
                                    if fc2 < base:
                                        base = fc2
                                        best_step = ts - x[d]
                                    break
                                step *= 0.6
                        if best_step != 0.0:
                            x[d] += best_step
                            x = clip_box(x)
                            fx = base
                            improved = True
                            if fx < best_f:
                                best_f = fx; best_x = x.copy()
                    if not improved:
                        break

        # Final clamp
        if best_x is not None:
            best_x = clip_box(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
