"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        # CMA-ES Parameters
        # lambda is population size, usually 4 + floor(3*log(n))
        lam = 4 + int(3 * np.log(self.dim))
        # Initial sigma, roughly 1/3 of the range
        sigma = 1.0
        # Mean
        m = np.zeros(self.dim)
        # Covariance matrix
        C = np.eye(self.dim)
        # Inverse square root of C for sampling
        # We use Cholesky decomposition for efficiency
        C_inv_sqrt = np.linalg.cholesky(C).T
        
        # Adaptation parameters
        mu = lam // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        d_sigma = c_sigma + 2 * (mu_eff / self.dim - 1) * (c_sigma + c_c)

        # State variables
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        # Initial evaluation
        x0 = m.copy()
        fx0 = func(x0)
        calls_made += 1
        best_f = fx0
        best_x = x0.copy()

        # Main Loop
        while calls_made < self.budget:
            # Check for stagnation or convergence
            if sigma < 1e-12:
                break
                
            # Generate population
            Y = np.zeros((lam, self.dim))
            for i in range(lam):
                z = rng.standard_normal(self.dim)
                Y[i] = m + sigma * (C_inv_sqrt @ z)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                # Clip to bounds? The problem statement says func clips, but 
                # it's safer to let func handle it or clip ourselves if we want to stay strictly inside.
                # The contract says "func(x) clips a finite length-5 vector to the box".
                # So we can just pass Y.
                F[i] = func(Y[i])
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
                
                # Stop if budget exceeded
                if calls_made >= self.budget:
                    break
            
            if calls_made >= self.budget:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update mean
            m_new = m + sigma * np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update path variables
            # We need C_inv_sqrt to compute p_sigma update correctly in some formulations, 
            # but standard CMA-ES uses:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            # Note: C_inv_sqrt @ (m_new - m) / sigma is the step in the "standard normal" space
            step = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            # Heaviside function for h_sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam))) < (1.4 + 2 / (self.dim + 1)) * np.sqrt(mu_eff + 1 - 1) else 0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(Y[:mu] - m, Y[:mu] - m), axis=0)
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive semi-definite (symmetrize and check eigenvalues if needed, 
            # but for small dim and short runs, simple symmetrization is usually enough)
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Recompute C_inv_sqrt
            try:
                C_inv_sqrt = np.linalg.cholesky(C).T
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2.0
                C_inv_sqrt = np.linalg.cholesky(C).T
            
            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
