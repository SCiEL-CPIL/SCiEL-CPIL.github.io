"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weighted parents
        # weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        # step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # covariance matrix adaptation parameters
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        # initial step size
        sigma = 2.0

        # initial mean
        rng = np.random.default_rng(0)
        m = rng.uniform(-5, 5, dim)

        # state variables
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        calls_made = 0

        # Evaluate initial mean
        fx = func(m)
        calls_made += 1
        best_x = m.copy()
        best_f = fx

        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                pop[i] = y
                if calls_made + 1 >= budget:
                    # Last call
                    pop_f[i] = func(y)
                    calls_made += 1
                else:
                    pop_f[i] = func(y)
                    calls_made += 1
                    if time.time() > deadline:
                        # Fill remaining with best so far to avoid exception
                        for j in range(i + 1, lambda_):
                            pop[j] = best_x
                            pop_f[j] = best_f
                        break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = pop[order]
            pop_f_sorted = pop_f[order]

            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of top mu individuals
            m_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)

            # Update evolution paths
            # Need to compute C^{-1/2} for ps update
            # C = B diag(D^2) B^T
            # C^{-1/2} = B diag(1/D) B^T
            C_inv_sqrt = B @ (np.diag(1.0 / D)) @ B.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (m_new - m) / sigma
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lambda_))) / np.sqrt(1 - c_sigma) < 1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma

            # Update covariance matrix
            delta_hs = (1 - hs) * c_c * (1 - c_c)
            C = B @ (np.diag(D**2)) @ B.T
            rank_mu = np.sum(w[:, None] * np.outer(pop_sorted[:mu] - m, pop_sorted[:mu] - m), axis=0)
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (np.outer(pc, pc) + delta_hs * C) + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            B = eigvecs
            D = np.sqrt(eigvals)

            m = m_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - c_sigma**2) - np.sqrt(dim)))
            sigma = max(sigma, 1e-10)

            # Check for termination
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
