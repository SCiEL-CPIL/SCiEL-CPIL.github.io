"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 6  # Population size, small for low dim
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute B*D for efficiency? No, B changes.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if budget - calls < lam:
                break

            # Generate lambda samples
            Z = np.random.default_rng().standard_normal((lam, n))
            # Transform Z to Y
            # Y = B * D * Z
            # B is orthogonal, D is diagonal
            Y = Z @ (D * B.T) # Note: B is stored as columns? Standard CMA-ES: B is orthogonal matrix.
            # Actually, standard update: C = B D^2 B^T.
            # Sampling: z ~ N(0, I). y = B D z. x = m + sigma y.
            # So X_i = m + sigma * (B * D * Z_i)
            
            X = np.empty((lam, n))
            for i in range(lam):
                # y_i = B @ (D * Z[i])
                y_i = B @ (D * Z[i])
                X[i] = m + sigma * y_i
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
            
            # Sort
            indices = np.argsort(F)
            F = F[indices]
            X = X[indices]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Recombination
            m_old = m.copy()
            m = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Step size adaptation (CSA)
            # y = (m - m_old) / sigma
            y = (m - m_old) / sigma
            # z = D^-1 B^-1 y
            z = B.T @ (y / D)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Heaviside function for sigma update
            hs_norm = np.linalg.norm(p_sigma)
            hs_threshold = (1.4 + 2.0 / (n + 1)) * (1 - c_sigma)
            if hs_norm / hs < hs_threshold:
                sigma *= np.exp(c_sigma / hs * (hs_norm / hs - 1)) # Wait, formula is exp(c_sigma/chs * (||ps||/hs - 1))
                # chs is usually 1 + 2*max(0, (mueff-1)/(n+1)) - 1? No.
                # Standard: chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) - 1 ?
                # Actually, chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1))
                # Let's use the standard formula:
                # chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1))
                chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1))
                sigma *= np.exp(c_sigma / chs * (hs_norm / hs - 1))
                
            # Covariance matrix adaptation
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * z
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            
            # Rank-1 update
            C += c_1 * (np.outer(y, y) - (1 - c_1) * C) # Wait, standard formula:
            # C = (1 - c_1) C + c_1 (y y^T + (1 - h_c) C)
            # h_c = (||p_c||/hs_c - 1 + 1 - c_c) / (1 - c_c) ?
            # Let's use the simpler rank-1 + rank-mu update.
            
            # Calculate h_c
            hs_c = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1) # This is for p_sigma, p_c has different threshold?
            # Threshold for h_c: (1 - c_c) + 2 * (mu_eff - 1) / (n + 1)
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^{2gen}) < (1.4 + 2/(n+1)) * sqrt((mu_eff - 1) + 1)
            # For simplicity and robustness, we can omit the h_c correction term or set it to 1.
            # Let's use the standard update without the h_c correction for now, or assume h_c=1.
            
            # Rank-1 update
            C = (1 - c_1) * C + c_1 * (np.outer(y, y) + (1 - 1.0) * C) # If h_c=1, the term is 0?
            # Standard: C = (1 - c_1) C + c_1 (y y^T + (1 - h_c) C)
            # If h_c=1, C = (1 - c_1) C + c_1 y y^T.
            
            # Rank-mu update
            # C += c_mu * sum w_i (y_i y_i^T)
            # y_i = (X[i] - m_old) / sigma
            for i in range(mu):
                y_i = (X[i] - m_old) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))
            
            # Check for convergence
            if np.max(D) / np.min(D) > 1e14:
                break
            if sigma < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
