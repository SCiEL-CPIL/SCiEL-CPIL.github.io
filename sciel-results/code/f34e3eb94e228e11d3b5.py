"""Rewritten optimizer: multi-start CMA-ES for the Katsuura (f23) instance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with a global exploration phase and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, deadline, call_count, best):
        """Run a correct CMA-ES. Returns (call_count, best)."""
        n = self.dim
        x = np.array(x0, dtype=np.float64)
        sigma = float(sigma0)
        pop = max(4, int(4 + 3 * np.log(n)))

        # Step-size control parameters
        cm = 1.0 / (pop + 4.7)
        ds = 1.0 + 2.0 * max(0.0, (n - 1) / (pop + 2) - 1.0)
        cs = (pop + 1.7 - n) / (n + pop + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((pop + 1) / (n + 1) - 1.0) - 1.0) + cs
        chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # Covariance adaptation parameters
        mue = pop // 2
        w = np.log(np.arange(1, mue + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cmu = min(0.1, (pop - 2 + 1.0 / (pop + n)) / ((pop + 2) * (n + 2)))
        mu = cmu * mueff

        C = np.eye(n)
        Cinv = np.eye(n)
        Csqrt = np.eye(n)
        Cinv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        def update_sqrt(C):
            # C = C + (2*mu) * (deltaC)  ->  Csqrt via Cholesky
            Csym = 0.5 * (C + C.T)
            # ensure positive definiteness
            eigvals = np.linalg.eigvalsh(Csym)
            if np.min(eigvals) < 1e-12:
                Csym = Csym + (1e-12 - np.min(eigvals)) * np.eye(n)
            Csqrt = np.linalg.cholesky(Csym).T
            Cinv = np.linalg.inv(Csym)
            # Cinv_sqrt via Cholesky of Cinv
            eigvals2 = np.linalg.eigvalsh(Cinv)
            if np.min(eigvals2) < 1e-12:
                Cinv = Cinv + (1e-12 - np.min(eigvals2)) * np.eye(n)
            Cinv_sqrt = np.linalg.cholesky(Cinv).T
            return Csym, Csqrt, Cinv, Cinv_sqrt

        # evaluate starting point
        fx0 = float(func(np.clip(x, -5.0, 5.0)))
        call_count[0] += 1
        if fx0 < best[1]:
            best[0] = np.clip(x, -5.0, 5.0).copy()
            best[1] = fx0

        while call_count[0] < budget and time.time() < deadline:
            Y = np.random.standard_normal((pop, n))
            X = x + sigma * (Y @ Csqrt.T)
            F = np.empty(pop)
            for i in range(pop):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                F[i] = float(func(np.clip(X[i], -5.0, 5.0)))
                call_count[0] += 1
                if F[i] < best[1]:
                    best[0] = np.clip(X[i], -5.0, 5.0).copy()
                    best[1] = F[i]
            if call_count[0] >= budget or time.time() > deadline:
                break

            order = np.argsort(F)
            Xw = X[order]
            Fw = F[order]
            xold = x.copy()
            x = np.sum(w[:, None] * Xw[:mue], axis=0)

            # covariance adaptation
            y = (x - xold) / sigma
            norm_y = np.linalg.norm(y)
            if norm_y > 1e-30:
                y = y / norm_y
            # hsig
            if np.linalg.norm(Csqrt @ y) / np.sqrt(
                    1.0 - (1.0 - sigma) ** (2.0 / (call_count[0] + 1))) < \
                    (1.4 + 2.0 / (n + 1)) * chiN:
                hsig = 1.0
            else:
                hsig = 0.0

            pc = (1 - cm) * pc + cm * hsig * y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * damps) * y

            hsig = 1.0
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2) < \
                    (1.4 + 2.0 / (n + 1)) * chiN:
                C = (1 + mu * (2 - mu) - mu) * C \
                    + mu * (np.outer(pc, pc) * (1 + (2 - hsig) * cs * (1 - cs)) - C)
            else:
                C = (1 + mu * (2 - mu) - mu) * C \
                    + mu * (np.outer(pc, pc) - C)
            C, Csqrt, Cinv, Cinv_sqrt = update_sqrt(C)

            # step size control
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = min(sigma, 1e6)

            x = np.clip(x, -5.0, 5.0)

        return call_count[0], best

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before the 300s limit

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(12345)

        call_count = [0]
        best = [None, np.inf]

        def safe(x):
            return float(func(np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)))

        # ---- Phase 0: global exploration (stratified random) ----
        n_explore = 60
        for _ in range(n_explore):
            if call_count[0] >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, n)
            fx = safe(x)
            call_count[0] += 1
            if fx < best[1]:
                best[0] = x.copy()
                best[1] = fx

        # ---- Phase 1: multi-start CMA-ES ----
        starts = []
        if best[0] is not None:
            starts.append((best[0].copy(), 1.0))
        for _ in range(3):
            starts.append((rng.uniform(-4.0, 4.0, n), 1.5))

        remaining = total_budget - call_count[0]
        n_starts = len(starts)
        per_start = max(1, remaining // n_starts)

        for sp, sig in starts:
            if call_count[0] >= total_budget or time.time() > deadline:
                break
            before = call_count[0]
            call_count[0], best = self._cmaes(
                func, sp, sig, min(before + per_start, total_budget),
                deadline, call_count, best)
            if best[1] < 1e-6:
                break

        return best[0], best[1]
# EVOLVE-BLOCK-END
