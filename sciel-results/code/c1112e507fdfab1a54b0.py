"""Optimizer for BBOB f20 (Schwefel) on [-5,5]^5.

Population-based CMA-ES with cumulative step-size adaptation (CSA),
restarts on stagnation, and a wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 300.0
        safety = 10.0
        rng = np.random.default_rng(12345)

        n = self.dim
        lam = 4 + int(3.0 * np.log(n)) if n > 1 else 2
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 * (mu_eff - 2 + 2.0 / n)) / ((n + 2) ** 2 + mu_eff - 2)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        def time_ok():
            return (time.time() - start) < (deadline - safety)

        best_x = None
        best_f = np.inf
        calls = 0

        max_restarts = 6
        restart_calls = 40  # min calls before considering a restart
        stall_calls = 60    # no improvement within this many -> restart

        for _ in range(max_restarts):
            # initial point: random in box
            x = rng.uniform(-5.0, 5.0, n)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            if not time_ok():
                break

            sigma = 1.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            count_success = 0
            no_improve = 0
            prev_best = fx

            while calls < self.budget and time_ok():
                # generate population
                B = rng.standard_normal((lam, n))
                # whitening: C^{1/2} via eigen
                evals, evecs = np.linalg.eigh(C)
                evals = np.sqrt(np.clip(evals, 1e-12, None))
                Csqrt = evecs @ (evecs.T * evals)  # evecs @ diag(evals)
                pops = x + sigma * (B @ Csqrt.T)
                pops = np.clip(pops, -5.0, 5.0)

                # evaluate
                fs = np.empty(lam)
                for i in range(lam):
                    fs[i] = func(pops[i])
                    calls += 1
                    if fs[i] < best_f:
                        best_f, best_x = fs[i], pops[i].copy()
                    if not time_ok():
                        break
                if not time_ok():
                    break

                # sort by fitness
                order = np.argsort(fs)
                fs_sorted = fs[order]
                pops_sorted = pops[order]

                # track improvement
                if fs_sorted[0] < prev_best - 1e-9:
                    prev_best = fs_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                # mean update
                y = pops_sorted[:mu]
                y_mean = (w[:, None] * y).sum(axis=0)
                y_norm = np.linalg.norm(y_mean)
                c_sigma_eff = c_sigma * d_sigma
                ps = (1 - c_sigma_eff) * ps + np.sqrt(c_sigma_eff * (2 - c_sigma_eff) * mu_eff) * y_norm / sigma * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / (n + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * y_mean / sigma
                # covariance update
                v = 1.0 / sigma
                diff = y * v
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (pc @ pc.T + (hsig - 1) * c_c * (2 - c_c) * C)
                     + c_mu * (diff.T @ (w[:, None] * diff)))

                # step size
                sigma *= np.exp(c_sigma_eff * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e6)

                x = x + y_mean
                x = np.clip(x, -5.0, 5.0)

                if no_improve > stall_calls:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
