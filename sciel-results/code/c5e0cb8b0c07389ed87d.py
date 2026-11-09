"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).

    Strategy:
      Phase 1: random sampling; build a covariance estimate from the
               principal components of the best early samples (PCA) so the
               search distribution aligns with the narrow funnel quickly.
      Phase 2: a single correct CMA-ES (covariance matrix adaptation) seeded
               with that covariance, adapting step-size and anisotropy.
      Phase 3: restart from the best point with a smaller sigma, then a final
               local refinement with small perturbations.

    Entry point and return interface are kept exactly as required.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---------------- Phase 1: random sampling + PCA covariance ----------------
        n_samples = min(60, max(10, total_budget // 8))
        samples = []
        for _ in range(n_samples):
            if time.time() > deadline - 2.0:
                break
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                continue
            calls += 1
            samples.append((f, x.copy()))
            record(x, f)

        # Build covariance from the best early samples (PCA alignment with funnel)
        cov = np.eye(dim)
        if len(samples) >= 3:
            samples_sorted = sorted(samples, key=lambda t: t[0])
            n_top = max(4, min(len(samples_sorted), len(samples_sorted) // 2))
            top = np.array([s[1] for s in samples_sorted[:n_top]])
            if top.shape[0] > 1:
                c = np.cov(top, rowvar=False)
                if c.size == 0 or not np.all(np.isfinite(c)):
                    c = np.eye(dim)
                c = np.atleast_2d(c)
                eigvals, eigvecs = np.linalg.eigh(c)
                eigvals = np.maximum(eigvals, 1e-6)
                # scale the anisotropy to a sensible magnitude
                trace = np.trace(c)
                scale = math.sqrt(trace / dim) if trace > 0 else 1.0
                cov = (eigvecs * eigvals) @ eigvecs.T
                cov = cov * (scale ** 2 / max(np.trace(cov), 1e-12))
                cov = (cov + cov.T) / 2

        # ---------------- CMA-ES core (reusable) ----------------
        pop_size = max(4, int(4 + 3 * math.log(dim)))
        lam = pop_size
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cmaes_run(budget_for_run, init_mean, init_sigma, init_cov):
            nonlocal calls
            mean = np.clip(init_mean.copy(), -5, 5)
            sigma = max(init_sigma, 1e-6)
            C = np.atleast_2d(init_cov.copy())
            if C.shape != (dim, dim):
                C = np.eye(dim)
            C = (C + C.T) / 2
            # ensure PSD
            try:
                ev, evv = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-10)
                C = (evv * ev) @ evv.T
                C = (C + C.T) / 2
            except Exception:
                C = np.eye(dim)
            try:
                B, D = _eigh_sqrt(C)
            except Exception:
                B, D = np.eye(dim), np.ones(dim)

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            gen = 0
            stall = 0
            local_best_f = float('inf')
            local_best_x = mean.copy()

            while calls < budget_for_run:
                if time.time() > deadline - 5.0:
                    break

                offspring = np.zeros((lam, dim))
                fitness = np.full(lam, np.inf)
                for i in range(lam):
                    if calls >= budget_for_run or time.time() > deadline - 2.0:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring[i] = y
                    try:
                        f = func(y)
                    except Exception:
                        f = np.inf
                    calls += 1
                    fitness[i] = f
                    record(y, f)
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = y.copy()

                if calls >= budget_for_run:
                    break

                order = np.argsort(fitness)
                s_off = offspring[order]
                s_fit = fitness[order]

                if s_fit[0] < local_best_f - 1e-12:
                    stall = 0
                else:
                    stall += 1
                if stall > 30 and gen > 5:
                    break

                # weighted recombination
                y_w = np.sum(w[:, None] * s_off[:mu], axis=0)
                mean_old = mean.copy()
                mean = mean + y_w
                mean = np.clip(mean, -5, 5)

                # step-size control (CSA)
                try:
                    y_sigma = (B @ (D * ((B.T @ (mean - mean_old)) / sigma))) / sigma
                    ps = (1 - c_sigma) * ps + \
                        math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_sigma
                    hsig = (np.linalg.norm(ps) /
                            math.sqrt(max(1 - (1 - c_sigma) ** (gen + 1) / d_sigma, 1e-12))) \
                        < (1.4 + 2 / (dim + 1)) * chi_n
                    sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                except Exception:
                    hsig = 1
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-10, 1e6))

                # covariance matrix adaptation
                try:
                    yc = (mean - mean_old) / sigma
                    pc = (1 - c_c) * pc + hsig * math.sqrt(c_c * (2 - c_c) * mu_eff) * yc
                    C = (1 + c1 * (hsig - 1) * c_c * (2 - c_c) - c1 - cmu) * C
                    C += c1 * np.outer(pc, pc)
                    C += cmu * np.sum(
                        w[:, None, None] * np.outer(s_off[:mu] - mean_old, s_off[:mu] - mean_old)
                        / sigma ** 2, axis=0)
                    C = (C + C.T) / 2
                    ev, evv = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-12)
                    C = (evv * ev) @ evv.T
                    C = (C + C.T) / 2
                    B, D = _eigh_sqrt(C)
                except Exception:
                    C = np.eye(dim)
                    B, D = np.eye(dim), np.ones(dim)
                    sigma = max(sigma * 0.5, 1e-6)

                # safety reset
                if sigma > 100 or sigma < 1e-9:
                    sigma = 1.0
                    C = np.eye(dim)
                    B, D = np.eye(dim), np.ones(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    mean = best_x.copy()

                gen += 1

            return local_best_x, local_best_f

        # ---------------- Phase 2: main CMA-ES seeded with PCA covariance ----------------
        budget1 = int(total_budget * 0.7)
        cmaes_run(budget1, best_x, 1.5, cov)

        # ---------------- Phase 3: restart from best with smaller sigma ----------------
        if calls < total_budget - 20 and time.time() < deadline - 10:
            budget2 = min(total_budget - calls, int(total_budget * 0.25))
            if budget2 > 10:
                cmaes_run(budget2, best_x, 0.5, np.eye(dim))

        # ---------------- Phase 4: local refinement ----------------
        if calls < total_budget - 5 and time.time() < deadline - 5:
            remaining = total_budget - calls
            for _ in range(min(remaining, 25)):
                if time.time() > deadline - 2:
                    break
                for scale in [0.1, 0.05, 0.01]:
                    if calls >= total_budget:
                        break
                    x_try = best_x + scale * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    try:
                        f_try = func(x_try)
                    except Exception:
                        calls += 1
                        continue
                    calls += 1
                    record(x_try, f_try)

        return best_x, best_f


def _eigh_sqrt(C):
    ev, evv = np.linalg.eigh(C)
    ev = np.maximum(ev, 1e-12)
    return evv, np.sqrt(ev)
# EVOLVE-BLOCK-END
