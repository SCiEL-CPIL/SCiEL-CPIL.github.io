"""Rewritten optimizer for BBOB f8 (Rosenbrock): CMA-ES basin search + path refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cmaes(func, dim, x0, sigma0, max_evals, deadline, x_lo, x_hi, rng,
           max_sigma=1e1, min_sigma=1e-7):
    """CMA-ES with box clipping and time guard. Returns (best_x, best_f, used)."""
    n = dim
    pop = max(4, int(round(4 + 3 * np.log(n))))
    c = 1.0 / (n + 4)
    d = 1.0 + 2 * max(0.0, np.sqrt((n - 1) / (n + 3)) - 1) / (n + 2)
    cm = 1.0 + c + d - 1.0 / n
    mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mu_eff = 1.0 / np.sum(w ** 2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    B = np.eye(n)
    D = np.ones(n)
    ps = np.zeros(n)
    pc = np.zeros(n)
    C = np.eye(n)
    x_best = x0.copy()
    f_best = func(x0)
    used = 1
    sigma = sigma0
    while used < max_evals:
        if time.time() > deadline:
            break
        sigma = float(np.clip(sigma, min_sigma, max_sigma))
        ys = [x_best + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(pop)]
        ys = [np.clip(y, x_lo, x_hi) for y in ys]
        fs = np.array([func(y) for y in ys])
        used += pop
        order = np.argsort(fs)
        yw = np.array(ys[order])
        fw = fs[order]
        mean = (w * yw).sum(axis=0)
        diff = yw - x_best
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B.T @ (D * (w @ diff)))
        if np.linalg.norm(ps) < (1.4 + 1 / c_sigma) * chi_n:
            pc = (1 - c_c) * pc + c_c * (mean - x_best) / sigma
        delta_h = 1 - (1 - c_c) ** max(max_evals - used, 1)
        C = ((1 - c_1 - c_mu + c_1 * c_mu * (1 - delta_h) * (np.dot(ps, ps) / n - 1)) * C
             + c_1 * (np.outer(pc, pc) - delta_h * (C * (pc[:, None] * pc[None, :])).sum(axis=1, keepdims=True))
             + c_mu * (w[:, None] * diff) @ (w[:, None] * diff).T)
        U, svals, Vt = np.linalg.svd(C)
        svals = np.clip(svals, 1e-12, None)
        B = U @ np.diag(1.0 / np.sqrt(svals)) @ Vt
        D = np.sqrt(svals)
        x_best = mean
        f_best = fw[0]
        sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
    return x_best, f_best, used


class Optimizer:
    """CMA-ES basin search + Rosenbrock path/coordinate refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        x_lo = np.full(dim, -5.0)
        x_hi = np.full(dim, 5.0)
        rng = np.random.default_rng(0)
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0
        used = 0
        x_best = None
        f_best = np.inf

        def evaluate(x):
            nonlocal used
            fx = float(func(np.asarray(x)))
            used += 1
            return fx

        # Phase 1: CMA-ES to locate the basin
        x0 = rng.uniform(-3, 3, dim)
        x_best, f_best, used1 = _cmaes(
            lambda x: evaluate(x), dim, x0, 2.0,
            max_evals=min(budget, 200), deadline=deadline,
            x_lo=x_lo, x_hi=x_hi, rng=rng)
        used += used1

        # Phase 2: Rosenbrock path refinement (x_{i+1} = x_i^2) in current frame
        def refine(x):
            nonlocal f_best, x_best
            base = x.copy()
            for _ in range(20):
                if used >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim - 1):
                    if used >= budget or time.time() > deadline:
                        break
                    cand = base.copy()
                    cand[i + 1] = base[i] ** 2
                    cand = np.clip(cand, x_lo, x_hi)
                    fc = evaluate(cand)
                    if fc < f_best:
                        f_best = fc
                        x_best = cand.copy()
                        base = cand.copy()
                        improved = True
                        # 1D polish on x_i
                        lo, hi = base[i] - 0.5, base[i] + 0.5
                        for _ in range(10):
                            if used >= budget or time.time() > deadline:
                                break
                            m1 = lo + (hi - lo) / 3
                            m2 = hi - (hi - lo) / 3
                            c1 = base.copy(); c1[i] = m1; c1 = np.clip(c1, x_lo, x_hi)
                            c2 = base.copy(); c2[i] = m2; c2 = np.clip(c2, x_lo, x_hi)
                            f1 = evaluate(c1); f2 = evaluate(c2)
                            if f1 < f2:
                                hi = m2
                            else:
                                lo = m1
                            if abs(hi - lo) < 1e-4:
                                break
                            base[i] = (lo + hi) / 2
                            if used >= budget or time.time() > deadline:
                                break
                        fc = evaluate(base)
                        if fc < f_best:
                            f_best = fc
                            x_best = base.copy()
                if not improved:
                    break
        refine(x_best.copy())

        # Phase 3: final CMA-ES polish
        if used < budget - 5:
            x_best, f_best, used2 = _cmaes(
                lambda x: evaluate(x), dim, x_best, 0.2,
                max_evals=budget - used, deadline=deadline,
                x_lo=x_lo, x_hi=x_hi, rng=rng)
            used += used2

        if x_best is None:
            x_best = np.zeros(dim)
        return x_best, f_best
# EVOLVE-BLOCK-END
