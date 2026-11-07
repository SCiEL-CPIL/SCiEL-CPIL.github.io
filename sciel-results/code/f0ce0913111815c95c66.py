"""Rewritten optimizer: CMA-ES with multi-start for Lunacek bi-Rastrigin (f24)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (covariance matrix adaptation) with deterministic multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin under the 330 s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---- deterministic multi-start seeds spread over the box ----
        rng0 = np.random.default_rng(12345)
        n_starts = 6
        starts = []
        for s in range(n_starts):
            if s == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng0.uniform(-4.0, 4.0, n)
            starts.append(np.clip(x0, -5.0, 5.0))

        # CMA-ES hyperparameters
        lambda_ = 4 + int(3.0 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 2.0 - mu - 0.5 * (lambda_ - 2 * mu) / (mu - 1.0)
        c_sigma = (mu + 0.5) / (n + 4.0 + mu)
        c_c = (4.0 + mu / n) / (n + 4.0 + 4.0 * mu / n)
        c_1 = (2.0 - nu * c_sigma - mu / n) / ((n + 1.5) ** 2 + mu)
        c_mu = min(1.0 - c_1,
                   (2.0 * (nu / 2.0 - 1.0) - c_1) /
                   ((n + 2.0) ** 2 + mu))

        for x_start in starts:
            if not time_ok():
                break
            x_mean = x_start.copy()
            f_mean = func(x_mean)
            update_best(x_mean, f_mean)

            # initial step size: large enough to cross Rastrigin oscillations
            sigma = 2.0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            sqrtC = np.eye(n)
            sqrtCinv = np.eye(n)
            sqrtc_1 = np.sqrt(c_1 * (n + 1.3) - 1.0)
            sqrtc_mu = np.sqrt(c_mu * (n + 2.0) - 1.0)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            sigma_min = 1e-4
            sigma_max = 5.0

            while budget > 0 and time_ok():
                # generate population
                ys = np.empty((lambda_, n))
                fs = np.empty(lambda_)
                for i in range(lambda_):
                    if not time_ok():
                        break
                    y = x_mean + sigma * (sqrtC @ rng0.standard_normal(n))
                    y = np.clip(y, -5.0, 5.0)
                    ys[i] = y
                    fs[i] = func(y)
                    budget -= 1
                    update_best(y, fs[i])
                if budget <= 0 or not time_ok():
                    break

                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # update mean
                x_mean = (weights[:, None] * ys_sorted[:mu]).sum(axis=0)
                f_mean = fs_sorted[0]
                update_best(x_mean, f_mean)

                # evolution paths
                diff = (ys_sorted[:mu] - x_mean) / sigma
                pc = (1.0 - c_sigma) * pc + c_sigma * np.sqrt(nu) * (weights @ diff)
                norm_pc = np.linalg.norm(pc)
                hsig = (norm_pc / np.sqrt(1.0 - (1.0 - c_sigma) ** (budget + 1))
                        < (1.4 + 2.0 / (n + 1)) * chi_n)
                ps = (1.0 - c_sigma) * ps + float(hsig) * np.sqrt(c_sigma * (2.0 - c_sigma) * nu) * diff
                ps = ps / (np.linalg.norm(ps) + 1e-18)

                # update covariance
                C = ((1.0 - c_1 - c_mu) * C
                     + c_1 * (np.outer(pc, pc) + (1.0 - float(hsig)) * nu * C)
                     + c_mu * (weights[:, None] * diff) @ (weights[:, None] * diff).T)
                C = 0.5 * (C + C.T)

                # sigma adaptation
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = min(max(sigma, sigma_min), sigma_max)

                # recompute sqrt(C) and inv(sqrt(C))
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.clip(evals, 1e-12, None)
                    sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                    sqrtCinv = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    sqrtC = np.eye(n)
                    sqrtCinv = np.eye(n)
                    break

                # stop if sigma is too small
                if sigma < sigma_min:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
