# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES for the BBOB f13 (Sharp ridge) black-box task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, rng, x0, sigma0, budget, deadline):
        n = self.dim
        x = np.asarray(x0, dtype=float)
        fx = float(func(x))
        best_x, best_f = x.copy(), fx

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.3) ** 2 + mueff)) * (2.0 - cc - (mueff - 2.0) / (n + mueff))
        cmu = min(
            1.0 - c1,
            2.0 * ((mueff - 2.0 + 2.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            * (1.0 + cc + (mueff / n - 1.0) / (n + 2.0)),
        )
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        sigma = float(sigma0)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        Cinv_sqrt = np.eye(n)
        Cinv_sqrt_inv = np.eye(n)
        gen = 0

        calls = 1
        while calls < budget:
            if time.time() > deadline:
                break

            # sample population
            ys = np.empty((lambda_, n))
            fs = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                zi = rng.standard_normal(n)
                y = x + sigma * (Cinv_sqrt @ zi)
                ys[i] = y
                fs[i] = float(func(y))
                calls += 1
            if calls >= budget:
                break

            # order by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # mean update
            m_old = x.copy()
            x = sum(weights[i] * ys[i] for i in range(mu))

            # evolution paths
            pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (Cinv_sqrt_inv @ (x - m_old) / sigma)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (x - m_old) / sigma

            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / chi_n < (1.4 + 2.0 / (n + 1.0))

            # covariance matrix update
            C = (
                (1.0 - c1 - cmu) * C
                + c1 * (pc @ pc + (1.0 - hsig) * cc * (2.0 - cc) * C)
                + cmu * sum(weights[i] * np.outer(ys[i] - m_old, ys[i] - m_old) for i in range(mu))
            )

            # sigma adaptation
            sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1.0))
            if sigma > 1e12:
                sigma = 1e12
            if sigma < 1e-12:
                break

            # recompute inverse square root of C (cheap for n=5)
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-12, None)
            Cinv_sqrt = (eigvec / np.sqrt(eigval)) @ eigvec.T
            Cinv_sqrt_inv = (eigvec * np.sqrt(eigval)) @ eigvec.T

            # stop if ill-conditioned / converged
            if (np.max(eigval) / np.min(eigval)) > 1e14:
                break
            if sigma < 1e-9:
                break
            gen += 1

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        n = self.dim
        # deadline guard: stop safely before the 330 s wall-clock kill
        deadline = time.time() + 280.0

        # deterministic restarts to be robust to the hidden shift/rotation
        n_restarts = 3
        per_budget = self.budget // n_restarts

        best_x = None
        best_f = np.inf
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                x0 = np.zeros(n)
            elif r == 1:
                x0 = 1.5 * rng.standard_normal(n)
            else:
                x0 = -1.5 * rng.standard_normal(n)
            x0 = np.clip(x0, -5.0, 5.0)
            sigma0 = 2.0
            xb, fb = self._cmaes(func, rng, x0, sigma0, per_budget, deadline)
            if fb < best_f:
                best_f = fb
                best_x = xb

        if best_x is None:
            x0 = np.zeros(n)
            best_f = float(func(x0))
            best_x = x0

        return best_x, best_f
# EVOLVE-BLOCK-END
