"""CMA-ES optimizer for BBOB f11 (Discus) with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_single(self, func, rng, x0, sigma0, max_calls, deadline,
                      global_best_x, global_best_f):
        """
        Run a single CMA-ES instance.
        Returns (best_x, best_f, calls_used).
        """
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Strategy parameters
        cc = (4 + mueff / n) / ((n + 4) + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc + cs)
        cmu = min(2 * (mueff - 1) * (0.5 + cs / cc) / ((n + 2) ** 2 + mueff),
                  c1 * (3 - cc))

        # State
        xmean = np.array(x0, dtype=float)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        # Cholesky factors
        B, D, Dinv = self._chol_update(C)

        chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        best_x = np.array(global_best_x, dtype=float)
        best_f = float(global_best_f)
        calls = 0
        stagnation_counter = 0
        best_f_history = best_f

        while calls < max_calls:
            if time.time() > deadline:
                break

            # Sample population
            z = rng.standard_normal((lam, n))
            pops = xmean[None, :] + sigma * (z @ B.T)
            pops = np.clip(pops, -5, 5)

            # Evaluate
            vals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break
                calls += 1
                try:
                    vals[i] = float(func(pops[i]))
                except Exception:
                    vals[i] = np.inf

            if calls >= max_calls:
                break

            # Update best
            order = np.argsort(vals)
            for i in order:
                if vals[i] < best_f:
                    best_f = vals[i]
                    best_x = pops[i].copy()

            # Selection
            idx = order[:mu]
            y = (pops[idx] - xmean[None, :]) / sigma
            yw = (w[:, None] * y).sum(axis=0)
            xnew = xmean + sigma * yw

            # Step-size control (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Dinv @ (B.T @ yw))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (n + cs)))) < \
                   ((1 + 2 / (n + 1)) * chiN)
            pc = (1 - cc) * pc + hsig * cc * yw

            # Covariance update
            yw_mat = yw[None, :]
            rank1 = np.outer(yw_mat[0], yw_mat[0])
            rankmu = (w[:, None, None] * np.einsum('ij,ik->ijk', y, y)).sum(axis=0)
            C = ((1 + c1 * cc * (cc - 1) - c1 - cmu) * C +
                 c1 * (rank1 + (cc - cc * cc) * C) +
                 cmu * rankmu)
            C = 0.5 * (C + C.T)

            # Cholesky update
            B, D, Dinv = self._chol_update(C)

            # Update mean
            xmean = xnew

            # Sigma update
            sigma = sigma * np.exp(cs / (5 * n) * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            # Termination check
            if norm_ps < 1e-10 and sigma < 1e-8:
                break

            # Stagnation detection
            if best_f < best_f_history - 1e-10:
                best_f_history = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnating for too long, break to allow restart
            if stagnation_counter > 2 * lam:
                break

        return best_x, best_f, calls

    def _chol_update(self, C):
        """Compute Cholesky factorization with fallback."""
        try:
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B @ B.T))
            Dinv = 1.0 / D
            return B, D, Dinv
        except np.linalg.LinAlgError:
            C = C + 1e-12 * np.eye(C.shape[0])
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T))
                Dinv = 1.0 / D
                return B, D, Dinv
            except np.linalg.LinAlgError:
                n = C.shape[0]
                return np.eye(n), np.ones(n), np.ones(n)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        total_budget = self.budget
        n = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Adaptive restart strategy
        # Start with a few diverse initial points
        num_starts = 5
        starts = [np.zeros(n)]
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, n))

        # Allocate budget: first start gets 40%, others split remaining
        b1 = int(total_budget * 0.4)
        remaining = (total_budget - b1) // (num_starts - 1)
        budgets = [b1] + [remaining] * (num_starts - 1)
        # Adjust last budget to use all calls
        budgets[-1] += total_budget - sum(budgets)

        for si, (x0, b) in enumerate(zip(starts, budgets)):
            if calls_used >= total_budget:
                break
            if time.time() > deadline:
                break

            sigma0 = 2.0 if si == 0 else 1.5
            bx, bf, c = self._cmaes_single(
                func, rng, x0, sigma0, b, deadline,
                best_x if best_x is not None else x0,
                best_f if np.isfinite(best_f) else np.inf
            )
            calls_used += c
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

            # If we found a good solution, do a polish phase
            if best_x is not None and calls_used < total_budget and time.time() < deadline:
                remaining = total_budget - calls_used
                # Polish with small sigma
                for sigma_p in [0.1, 0.01, 0.001]:
                    if calls_used >= total_budget or time.time() > deadline:
                        break
                    b_polish = min(remaining, 30)
                    bx, bf, c = self._cmaes_single(
                        func, rng, best_x, sigma_p, b_polish, deadline,
                        best_x, best_f
                    )
                    calls_used += c
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()

        # Final jitter refinement
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            jitter_sigma = 1e-3
            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls_used >= total_budget:
                    break
                cand = np.clip(best_x + jitter_sigma * rng.standard_normal(n), -5, 5)
                calls_used += 1
                try:
                    cf = float(func(cand))
                except Exception:
                    cf = np.inf
                if cf < best_f:
                    best_f = cf
                    best_x = cand.copy()
                    jitter_sigma *= 0.5
                else:
                    jitter_sigma *= 0.7
                if jitter_sigma < 1e-8:
                    break

        if best_x is None:
            best_x = np.zeros(n)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
