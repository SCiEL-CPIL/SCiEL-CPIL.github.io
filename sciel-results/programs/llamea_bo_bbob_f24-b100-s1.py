"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def _make_func(self, func):
        """Wrap func to count calls."""
        def wrapped(x):
            self.call_count += 1
            return float(func(x))
        return wrapped

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.call_count = 0
        
        # Bounds
        lb = -5.0
        ub = 5.0
        
        # Create a counting wrapper
        wrapped_func = self._make_func(func)
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        # For dim=5, lam = 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for better exploration.
        lam = 10
        
        # Weighting
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initialization
        rng = np.random.default_rng(42)
        m = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = None
        best_f = np.inf
        
        while self.call_count < budget:
            # Sample population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform to normal with covariance C
                # x = m + sigma * B * D * z
                # We maintain C = B D^2 B^T
                # For simplicity in small dim, we can just use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    x = m + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x = m + sigma * z
                
                # Clip to bounds
                x = np.clip(x, lb, ub)
                pop[i] = x
            
            # Evaluate
            fits = np.array([wrapped_func(x) for x in pop])
            
            # Track best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()
                
            if self.call_count >= budget:
                break
                
            # Sort indices
            order = np.argsort(fits)
            pop_sorted = pop[order]
            fits_sorted = fits[order]
            
            # Update mean
            m_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update covariance matrix (simplified: use rank-one and rank-mu)
            # Actually, for dim=5, we can just re-estimate C from the best mu points?
            # No, CMA-ES updates C incrementally.
            # Let's implement the standard update.
            
            # Step size adaptation (CSA)
            pc = (1 - c_sigma) * pc + c_sigma * (m_new - m) / sigma
            norm_pc = np.linalg.norm(pc)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_pc / chi_n - 1))
            
            # Covariance matrix adaptation
            # Rank-one update
            pc_hat = pc / norm_pc if norm_pc > 1e-12 else np.zeros(dim)
            C = (1 + c_1 * (2 - c_c) - c_1) * C + c_1 * np.outer(pc_hat, pc_hat)
            
            # Rank-mu update
            z = np.zeros((mu, dim))
            for i in range(mu):
                z[i] = (pop_sorted[i] - m) / sigma
            C += c_mu * np.sum(weights[:, None, None] * z[:, :, None] * z[:, None, :], axis=0)
            
            # Ensure C is positive definite
            # Sometimes numerical issues arise. We can add a small regularization.
            C = 0.5 * (C + C.T)
            C += 1e-8 * np.eye(dim)
            
            m = m_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
