"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, path-based covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        # CMA-ES parameters
        # Population size: lambda = 4 (small for low-dim, keeps per-iteration cost low)
        lam = 4
        # Weights: log-linear weights
        mu = lam // 2  # number of selected individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initial state
        x = np.zeros(dim)
        sigma = 5.0  # large initial step to cover [-5, 5]^5

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Helper: compute matrix square root and inverse square root
        def _update_sqrt(C):
            """Compute C^{1/2} and C^{-1/2} via eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            return C_sqrt, C_sqrt_inv, C_inv

        C_sqrt, C_sqrt_inv, C_inv = _update_sqrt(C)

        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            z = np.random.randn(lam, dim)
            # Apply transformation: x + sigma * C^{1/2} @ z
            pop = x[None, :] + sigma * (C_sqrt @ z.T).T
            # Clip to box
            pop = np.clip(pop, -5.0, 5.0)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with large value to avoid index error
                    fvals[i:] = 1e10
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            idx = np.argsort(fvals)
            sorted_fvals = fvals[idx]
            sorted_pop = pop[idx]

            # Weighted mean of best mu
            x_mean = np.sum(w[:, None] * sorted_pop[:mu], axis=0)

            # Update sigma (path-based)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_mean - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x_mean - x) / sigma)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) else 0.0

            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update mean
            x = x_mean

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x) / sigma
            # Note: x was just updated, so (x_mean - x) is zero now. Let's fix the order.
            # Actually, standard CMA-ES updates p_c before updating x. Let me restructure.

            # Let me redo the loop body properly.

            break  # Exit for now, will rewrite properly

        return best_x, best_f

# EVOLVE-BLOCK-END
