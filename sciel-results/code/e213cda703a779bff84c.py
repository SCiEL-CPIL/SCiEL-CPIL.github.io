"""Optimizer for BBOB f13 (Sharp ridge) on [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES + coordinate descent for the sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin below the 300s limit

        # --- deterministic candidate set for the multi-start phase ---
        rng0 = np.random.default_rng(0)
        starts = [rng0.uniform(-5, 5, dim)]
        # a few spread-out deterministic starts to improve early coverage
        for s in range(1, 6):
            g = np.random.default_rng(100 + s)
            starts.append(g.uniform(-5, 5, dim))

        def cmaes(x0, sigma0, rng, max_calls, deadline):
            """(mu+lambda)-CMA-ES with adaptive sigma. Returns best (x, f)."""
            d = dim
            lam = 6
            mu = 3
            w = np.log(mu + 0.5) - np.log(1, ) if False else None
            # standard weights
            w = np.log(mu + 0.5) - np.arange(1, mu + 1)
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

            xmean = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(d)
            ps = np.zeros(d)
            B = np.eye(d)
            D = np.ones(d)
            C = np.eye(d)
            invsqrtC = np.eye(d)
            calls = 0
            best_x = xmean.copy()
            best_f = func(best_x)
            calls += 1
            if calls >= max_calls:
                return best_x, best_f

            def invsqrt(C):
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-20)
                return B, D

            while calls < max_calls and time.time() < deadline:
                # generate lambda offspring
                ys = np.empty((lam, d))
                xs = np.empty((lam, d))
                for i in range(lam):
                    z = rng.standard_normal(d)
                    # x = xmean + sigma * B D z
                    xs[i] = xmean + sigma * (B @ (D * z))
                    ys[i] = z
                # evaluate
                fs = np.empty(lam)
                for i in range(lam):
                    fs[i] = func(xs[i])
                    calls += 1
                    if calls >= max_calls:
                        break
                # handle early stop
                if calls >= max_calls:
                    # update best
                    idx = int(np.argmin(fs))
                    if fs[idx] < best_f:
                        best_x, best_f = xs[idx].copy(), fs[idx]
                    break

                # sort by fitness
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ys = ys[order]

                # update best
                if fs[0] < best_f:
                    best_x, best_f = xs[0].copy(), fs[0]

                # recombination
                y_w = sum(w[i] * ys[i] for i in range(mu))
                # update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_d * (invsqrtC @ y_w)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
                sigma = min(sigma, 5.0)
                # covariance adaptation
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (d + 1)) * chi_d
                v = (invsqrtC @ y_w) if hsig else 0.0
                pc = (1 - c_c) * pc + hsig * c_c * v
                rank_one = np.outer(pc, pc)
                rank_mu = sum(w[i] * np.outer(ys[i], ys[i]) for i in range(mu))
                C = (1 + c_c * (2 - c_c) * hsig - c_sigma) * C \
                    + c_sigma * (rank_one + (1 - hsig) * c_c * (2 - c_c) * rank_mu)
                # recompute B, D
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-20)
                invsqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                # update mean
                xmean = xmean + sigma * (B @ (D * y_w))

            return best_x, best_f

        def coord_desc(x, fx, max_calls, deadline):
            """Simple coordinate descent polish."""
            d = dim
            calls = 0
            best_x = np.array(x, dtype=float)
            best_f = fx
            step = 0.5
            while calls < max_calls and time.time() < deadline:
                improved = False
                for i in range(d):
                    for sign in (1.0, -1.0):
                        cand = best_x.copy()
                        cand[i] += sign * step
                        cand = np.clip(cand, -5, 5)
                        fc = func(cand)
                        calls += 1
                        if fc < best_f:
                            best_x, best_f = cand, fc
                            improved = True
                        if calls >= max_calls:
                            return best_x, best_f
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            return best_x, best_f

        # --- Phase 1: multi-start CMA-ES, distributing the budget ---
        # Give the first (best) start the bulk of the budget; others get a small probe.
        # We'll do a short probe on each start, then run CMA-ES on the best probe.
        probe_budget = min(30, budget // len(starts))
        probe_results = []
        used = 0
        for s_idx, s in enumerate(starts):
            if used + probe_budget > budget:
                probe_budget = max(1, budget - used)
            rng_p = np.random.default_rng(200 + s_idx)
            bx, bf = cmaes(s, 0.4, rng_p, probe_budget, deadline)
            probe_results.append((bf, bx))
            used += probe_budget
            if used >= budget:
                break

        # pick the best probe as the seed for the main CMA-ES
        probe_results.sort(key=lambda t: t[0])
        seed_x = probe_results[0][1].copy()
        seed_f = probe_results[0][0]

        # --- Phase 2: main CMA-ES from the best seed with the remaining budget ---
        remaining = budget - used
        if remaining > 5 and time.time() < deadline:
            rng_main = np.random.default_rng(7)
            bx, bf = cmaes(seed_x, 0.3, rng_main, remaining, deadline)
            if bf < seed_f:
                seed_x, seed_f = bx, bf

        # --- Phase 3: coordinate descent polish on the best point ---
        if time.time() < deadline:
            px, pf = coord_desc(seed_x, seed_f, 100, deadline)
            if pf < seed_f:
                seed_x, seed_f = px, pf

        return seed_x, seed_f
