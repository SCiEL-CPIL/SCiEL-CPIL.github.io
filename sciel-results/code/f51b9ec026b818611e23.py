"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for the Bent Cigar (f12) landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda: population size
        # For dim=5, a common choice is lambda = 4 + floor(3*log(dim)) = 4 + 3*1 = 7
        # But for small budgets, smaller lambda might be better. Let's use 4 + 3*log(dim)
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        # Ensure lambda is at least 2
        lam = max(lam, 2)

        # Mu: number of best solutions used for update
        mu = lam // 2
        # Weights for the mu best solutions
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Step size adaptation parameters
        # Standard CMA-ES: c_sigma, d_sigma, c_c, d_c, c_1, c_mu
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        # Start near the center of the box
        x = np.zeros(dim)
        sigma = 2.5  # Initial step size, roughly 1/2 of box width
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (c_c + 1) / 2  # For pc update
        # Actually, standard CMA-ES uses:
        # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * sigma * B^T * C^(-1/2) * (x_w - x)/sigma
        # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * B^T * C^(-1/2) * (x_w - x)/sigma

        # We need the inverse square root of C. We'll use Cholesky decomposition.
        # For dim=5, this is cheap.

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Precompute some constants
        # We'll evaluate the population each generation
        # Number of generations: budget / lam
        # But we might run out of budget mid-generation. We handle that.

        while calls < budget:
            # Generate population
            # We need to generate lam points, but we might not have budget for all
            # Let's generate as many as we can afford
            n_evals_this_gen = min(lam, budget - calls)
            if n_evals_this_gen < 2:
                # Can't do a proper CMA-ES update with < 2 evaluations
                # Just do a simple random search or stop
                # Let's just do a few random steps
                while calls < budget:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy
                break

            # Generate n_evals_this_gen points
            # For simplicity, let's just generate lam points and evaluate all
            # But we need to respect the budget. If budget - calls < lam, we can't do a full generation.
            # Strategy: if we can't do a full generation, do a final random search.
            if budget - calls < lam:
                # Do random search for remaining budget
                for _ in range(budget - calls):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy
                break

            # Generate population
            pop = np.zeros((lam, dim))
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            # L is lower triangular such that C = L @ L.T
            # We need C^(-1/2). We can use L_inv = inv(L), then C^(-1/2) = L_inv.T @ L_inv
            # But for generating samples, we can use: x_i = x + sigma * L.T @ z_i
            # where z_i ~ N(0, I). This is equivalent to x + sigma * C^(-1/2) @ z_i if we adjust.
            # Actually, if we want to sample from N(x, sigma^2 * C), we do:
            # x + sigma * L.T @ z, where z ~ N(0, I)
            # Because Cov(L.T @ z) = L.T @ I @ L = C
            # So this is correct.
            for i in range(lam):
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * L.T @ z
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fpop = np.zeros(lam)
            for i in range(lam):
                fpop[i] = func(pop[i])
                calls += 1

            # Update best
            min_idx = np.argmin(fpop)
            if fpop[min_idx] < best_f:
                best_f = fpop[min_idx]
                best_x = pop[min_idx].copy()

            # Sort population by fitness
            order = np.argsort(fpop)
            pop = pop[order]
            fpop = fpop[order]

            # Compute weighted mean
            x_w = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size (sigma)
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C^(-1/2) * (x_w - x) / sigma
            # We need C^(-1/2). Let's compute it.
            # C^(-1/2) = L^(-T) @ L^(-1)
            # Or we can use: C^(-1/2) = inv(sqrtm(C))
            # For dim=5, we can use scipy or manual.
            # Let's use the fact that C = L @ L.T, so C^(-1/2) = L^(-T) @ L^(-1)
            # We can compute L_inv = np.linalg.inv(L)
            L_inv = np.linalg.inv(L)
            C_inv_sqrt = L_inv.T @ L_inv

            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma

            # Update chi_norm
            # chi_norm = sqrt(dim) * (1 - 1/(4*dim) + 1/(16*dim**2))
            chi_norm = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (16 * dim ** 2))

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_norm - 1))

            # Update pc
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * C^(-1/2) * (x_w - x) / sigma
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma

            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (c_c - 2 + c_c^2) * sigma^2 * (np.linalg.norm(pc)**2 / dim - 1) * I)
            #    + c_mu * sum(wi * (yi - x) @ (yi - x).T)
            delta_sigma = (c_c - 2 + c_c ** 2) * sigma ** 2 * (np.linalg.norm(pc) ** 2 / dim - 1)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sigma * np.eye(dim))
            for i in range(mu):
                diff = pop[i] - x
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Update mean
            x = x_w

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Check for positive definiteness
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, add small diagonal
                C += 1e-8 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
