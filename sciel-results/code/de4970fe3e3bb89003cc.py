import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285
        rng = np.random.default_rng(42)
        
        n = dim
        # For Rosenbrock, larger lambda helps with the elongated valley
        lambda_ = 20
        mu = lambda_ // 2
        
        # Initial state
        m = np.zeros(n)
        sigma = 2.5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights (linear ranking)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Track best
        best_x = m.copy()
        best_f = func(m)
        calls = 1
        
        gen = 0
        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break
            
            # Compute C^(-1/2) via eigendecomposition (n=5 is tiny)
            C_sym = (C + C.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(C_sym)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample
            Z = rng.standard_normal((lambda_, n))
            X = m[None, :] + sigma * (Z @ C_inv_sqrt.T)
            
            # Evaluate
            F = np.empty(lambda_)
            for i in range(lambda_):
                x_i = np.clip(X[i], -5.0, 5.0)
                F[i] = func(x_i)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
            
            if time.time() > deadline:
                break
            
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update paths
            diff = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ diff)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)
            
            # h_sigma indicator
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 50.0)
            
            # Update covariance
            rank1_term = np.outer(p_c, p_c)
            if not h_sigma:
                rank1_term = rank1_term + (1 - (1 - c_c)**2) * C_sym
            
            D = X_sorted[:mu] - m_old[None, :]
            rank_mu = np.sum(weights[:, None, None] * D[:, :, None] * D[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C_sym + c_1 * rank1_term + c_mu * rank_mu
            C = (C + C.T) / 2.0
            
            gen += 1
        
        # Use remaining budget for local refinement around best point
        remaining = budget - calls
        if remaining > 5 and best_f < np.inf:
            # Simple local search: try small perturbations
            local_sigma = max(sigma * 0.1, 0.01)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                x_try = best_x + rng.standard_normal(n) * local_sigma
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = func(x_try)
                calls += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    local_sigma *= 0.7  # shrink on improvement
                else:
                    local_sigma *= 1.05  # slight expansion
        
        return best_x, best_f
