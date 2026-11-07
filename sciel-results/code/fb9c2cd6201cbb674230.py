"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based CMA-ES with an early random sweep and a wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _mat_sqrt(self, A):
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        return V @ (np.sqrt(w)[:, None] * V.T)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = 320.0  # safety margin before the 330s kill

        rng = np.random.default_rng(12345)

        # ---- Phase 0: cheap random sweep for a strong early best (AOCC) ----
        best_x = None
        best_f = np.inf
        n_sweep = 12
        for _ in range(n_sweep):
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        if time.time() - start_time > deadline:
            return best_x, best_f

        # ---- Phase 1: population-based CMA-ES ----
        mean = best_x.copy()
        sigma = 1.2
        lam = 12  # population size
        mu = lam // 2

        weights = np.log(np.arange(lam, 0, -1) + 1.0)
        weights = weights / weights.sum()
        w = weights[:mu]
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c + c_sigma)
        c_mu = (2 / ((dim + 2) ** 2 + mu_eff)) * min(
            1 - c_1, 2 / ((dim + 1.3) ** 2 + mu_eff)
        )
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        calls_made = n_sweep
        gen = 0
        while calls_made < budget:
            if time.time() - start_time > deadline:
                break
            # Generate population
            offspring = np.empty((lam, dim))
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() - start_time > deadline:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                offspring[i] = x
                fvals[i] = func(x)
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()
            if calls_made >= budget:
                break
            if time.time() - start_time > deadline:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            x_sorted = offspring[order]

            # Weighted mean of best mu
            new_mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)

            inv_sqrt_C = np.linalg.inv(C_sqrt)
            step = inv_sqrt_C @ (new_mean - mean) / sigma

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * step
            gen += 1
            hsig = (
                np.linalg.norm(ps)
                / np.sqrt(max(1e-12, 1 - (1 - c_sigma) ** (2 * gen)))
                < (1 + 2 / (dim + 1)) * chi_n
            )
            pc = (1 - c_c) * pc + float(hsig) * np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * step

            # Rank-1 + rank-mu covariance update
            C = (1 + c_1 + c_mu) * C - c_1 * C
            C = C + c_1 * np.outer(step, step)
            outer_sum = np.sum(
                w[:, None, None]
                * np.einsum("ij,ik->ijk", x_sorted[:mu], x_sorted[:mu]),
                axis=0,
            )
            C = C + c_mu * outer_sum
            C = 0.5 * (C + C.T)

            # Update sigma (CSA)
            sigma = sigma * np.exp(
                c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1)
            )
            sigma = float(np.clip(sigma, 0.05, 5.0))

            mean = new_mean
            C_sqrt = self._mat_sqrt(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
