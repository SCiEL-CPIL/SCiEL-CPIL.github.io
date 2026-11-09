"""CMA-ES based black-box optimizer for the rotated Sphere (BBOB f1)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with a small fixed population, adapted to a rotated 5-D sphere."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Population size (lambda) and its associated constants
        lam = 6
        mu = lam // 2                       # number of recombination weights

        # Recombination weights (standard CMA-ES scheme)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0.0, (mueff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial mean and sigma
        x = rng.uniform(-5.0, 5.0, d)
        sigma = 2.0

        # CMA-ES state
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        invsqrtC = np.eye(d)
        invsqrtCpow = 1.0
        Cpow = 1.0
        D = 1.0

        def sample(n):
            # Draw n standard-normal vectors, project through invsqrtC * D
            z = rng.standard_normal((n, d))
            return (z @ invsqrtC.T) * D

        def update_state(x_mean, x_weights, sigma):
            nonlocal C, pc, ps, invsqrtC, invsqrtCpow, Cpow, D
            # Eigen-decompose C (symmetric positive-definite)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            # Update inverse-sqrt and powers
            invsqrtC = (evecs / np.sqrt(evals)) @ evecs.T
            invsqrtCpow *= (1.0 + c_1) ** (1.0 / (2.0 * d))
            Cpow *= (1.0 + c_1 + c_mu) ** (1.0 / (2.0 * d))
            D = np.prod(np.sqrt(evals))
            # Cumulative step-size adaptation
            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) \
                 * (invsqrtC @ x_weights) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** 2) < \
                   (1.4 + 2.0 / (d + 1)) * chi_d
            pc = (1.0 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * x_weights / sigma
            # Update C
            delta = (1.0 + hsig * (c_c - 1.0) - c_c) * np.outer(pc, pc)
            C = (1.0 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (delta - C) \
                + c_mu * np.sum(w[:, None] * np.outer(x_weights, x_weights), axis=0)
            # Renormalise to keep determinant consistent (avoid drift)
            C = C / (np.linalg.det(C) ** (1.0 / d)) if np.linalg.det(C) > 0 else C
            C = 0.5 * (C + C.T)

        fx = func(x)
        calls = 1
        gen = 0
        while calls < budget:
            # Determine how many candidates we can afford this generation
            n_this = min(lam, budget - calls)
            if n_this < 2:
                break
            z = sample(n_this)
            x_pop = x + sigma * z
            x_pop = np.clip(x_pop, -5.0, 5.0)
            f_pop = np.empty(n_this)
            for i in range(n_this):
                f_pop[i] = func(x_pop[i])
                calls += 1
            # Order by fitness (best = smallest)
            order = np.argsort(f_pop)
            # Use the best mu (or fewer if we had a short generation)
            m = min(mu, n_this)
            idx = order[:m]
            # Recompute weights for m candidates
            wm = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
            wm = wm / wm.sum()
            mueff_m = 1.0 / np.sum(wm ** 2)
            x_weights = np.sum(wm[:, None] * x_pop[idx], axis=0)
            new_mean = x_weights
            # Step-size adaptation
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(x_weights) / chi_d - 1.0))
            sigma = np.clip(sigma, 1e-12, 50.0)
            # Update mean and covariance
            x = new_mean
            update_state(x, x_weights, sigma)
            gen += 1
        return x, fx
# EVOLVE-BLOCK-END
