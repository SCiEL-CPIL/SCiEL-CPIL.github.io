"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def time_ok():
            nonlocal calls_used
            return time.time() < deadline and calls_used < budget

        def wrapped(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(0)

        def run_cma(x0, sigma0, max_iter):
            nonlocal best_x, best_f, calls_used
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 1) ** 2 + mueff)
            cmu = (2 * (mueff - 2 + 2.0 / (n + mueff))) / ((n + 2) ** 2 + mueff)
            cs = (mueff + 1.5 - n / 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            sigma = sigma0
            sigma0_save = sigma
            # Cumulation vectors
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            # Cholesky factor
            try:
                Cchol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                Cchol = np.eye(n)
            # Mean
            m = np.array(x0, dtype=float)
            # Track best
            local_best_x = m.copy()
            local_best_f = wrapped(m)
            # Evaluation counter for this run
            evals_this_run = 1

            for gen in range(max_iter):
                if not time_ok():
                    break
                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(Cchol) * \
                     (np.linalg.solve(Cchol, m - m_old) if False else np.linalg.solve(Cchol, m - m_prev)) if False else (1 - cs) * ps
                # Recompute ps properly
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(Cchol) * \
                     np.linalg.solve(Cchol, m - m_prev)
                chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_this_run + 1))) > (1.4 + 2 / (n + 1)) * chiN:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))

                # Sample population
                # Update C (do it after we have new mean, but standard order: sample, evaluate, then update)
                # We need Cchol for sampling. Update C after evaluation.
                # Let's restructure: sample using current Cchol
                ys = np.zeros((lam, n))
                for i in range(lam):
                    ys[i] = np.linalg.solve(Cchol.T, rng.standard_normal(n))  # Cchol^{-T} z
                # Actually solve(Cchol, z) gives Cchol^{-1} z. We want C^{1/2} z = Cchol z.
                # C = Cchol @ Cchol.T. So C^{1/2} z = Cchol z.
                # Let's just use Cchol @ z
                ys = np.array([Cchol @ rng.standard_normal(n) for _ in range(lam)])
                pop = np.array([m + sigma * y for y in ys])
                # Clip to box
                pop = np.clip(pop, -5, 5)

                # Evaluate
                fits = np.array([wrapped(x) for x in pop])
                # Update best
                idx = np.argmin(fits)
                if fits[idx] < local_best_f:
                    local_best_f = fits[idx]
                    local_best_x = pop[idx].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                evals_this_run += lam

                # Sort by fitness
                order = np.argsort(fits)
                y_sorted = pop[order]
                f_sorted = fits[order]

                # New mean
                m_prev = m.copy()
                m_new = np.sum(w[:, None] * y_sorted[:mu], axis=0)
                m = m_new

                # Update C
                # Cumulation for C
                pc = (1 - cmu) * pc + np.sqrt(cmu * (2 - cmu) * mueff) / np.linalg.norm(Cchol) * \
                     np.linalg.solve(Cchol, m - m_prev)
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (evals_this_run + 1))) < \
                       (1.4 + 2 / (n + 1)) * chiN
                hsig = float(hsig)

                # Rank-mu update
                ys = (y_sorted[:mu] - m_prev) / sigma
                rank_mu = np.sum(w[:, None] * np.outer(ys.T, ys), axis=0)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(m - m_prev, m - m_prev) / sigma ** 2 +
                                                 (hsig - 1) * c1 * cs * (2 - cs) * np.outer(pc, pc)) + \
                    cmu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update Cholesky
                try:
                    Cchol = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    Cchol = np.eye(n)

                # Termination check
                if sigma < 1e-12 or sigma > 1e12:
                    break
                # Convergence
                if np.linalg.norm(m - m_prev) < 1e-10 * max(1, np.linalg.norm(m)):
                    break

            return local_best_x, local_best_f

        # Multi-start CMA-ES
        # First try from center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        max_iter = 200
        _, _ = run_cma(x0, sigma0, max_iter)

        if time_ok():
            # Second try from a random point
            x1 = rng.uniform(-4, 4, dim)
            _, _ = run_cma(x1, 1.5, 150)

        if time_ok():
            # Third try from another random point
            x2 = rng.uniform(-4, 4, dim)
            _, _ = run_cma(x2, 1.0, 100)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
