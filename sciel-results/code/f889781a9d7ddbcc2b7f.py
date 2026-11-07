"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    Uses CMA-ES with careful parameter settings and a time guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Initial point
        x0 = rng.uniform(-5, 5, n)
        best_x = x0.copy()
        best_f = func(x0)
        evals = 1

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        # Step size
        sigma = 1.0  # Start with reasonable step size for [-5,5]^5

        # Mean
        m = x0.copy()

        # Covariance matrix
        C = np.eye(n)

        # Path accumulators
        ps = np.zeros(n)
        pc = np.zeros(n)

        # CMA-ES adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

        # Precompute some constants
        hc = None  # Will be computed in loop

        gen = 0

        while evals < budget:
            if time.time() > deadline:
                break

            # Get C^(-1/2)
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = (B / np.sqrt(D)) @ B.T

            # Generate offspring
            Y = np.zeros((lam, n))
            F = np.empty(lam)

            for i in range(lam):
                if evals >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                Y[i] = m + sigma * (C_inv_sqrt @ z)
                F[i] = func(Y[i])
                evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()

            if evals >= budget or time.time() > deadline:
                break

            # Sort offspring
            order = np.argsort(F)
            Y_best = Y[order[:mu]]
            F_best = F[order[:mu]]

            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * Y_best, axis=0)

            # Update path statistics
            m_diff = m - m_old
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ m_diff)

            # Check for convergence of path
            gen += 1
            # Use a simpler h_c threshold
            hc = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) <
                  (1.4 + c_c / n) * np.sqrt((n + 1) / (n - 1)))

            pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ m_diff)

            # Update covariance matrix
            rank_one = pc @ pc.T + (1 - hc) * c_sigma * (2 - c_sigma) * C
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = C_inv_sqrt @ (Y_best[i] - m_old)
                rank_mu += w[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Early termination check: if sigma is very small, we've converged
            if sigma < 1e-6:
                break

            # If the best value hasn't improved in many generations, consider stopping
            # (optional, but CMA-ES usually handles this)

        return best_x, best_f
# EVOLVE-BLOCK-END
