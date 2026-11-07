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

        # Phase 1: Coarse Random Search to find a good starting point
        # Use 10% of budget for this phase
        coarse_budget = max(10, int(total_budget * 0.1))
        for _ in range(coarse_budget):
            if calls_used >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            calls_used += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES from the best point found
        if calls_used < total_budget and time.time() < deadline and best_x is not None:
            # Use remaining budget for CMA-ES
            remaining_budget = total_budget - calls_used
            # Start with a larger sigma to escape local minima if any
            sigma0 = 2.0
            bx, bf, c = self._cmaes_single(
                func, rng, best_x, sigma0, remaining_budget, deadline,
                best_x, best_f
            )
            calls_used += c
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Polish with smaller sigma if time/budget remains
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            # Try a few polish steps with decreasing sigma
            for sigma_p in [0.5, 0.1, 0.01]:
                if calls_used >= total_budget or time.time() > deadline:
                    break
                b_polish = min(remaining, 50)
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
