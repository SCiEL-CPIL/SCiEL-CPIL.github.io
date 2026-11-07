import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n)) # 8 for n=5
        mu = lam // 2                 # 4
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        m = rng.uniform(-0.5, 0.5, n) # Small random start
        C = np.eye(n)
        sigma = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = m.copy()
        best_f = func(m)
        budget -= 1
        
        while budget >= lam:
            # Sample
            Y = rng.multivariate_normal(np.zeros(n), C, size=lam)
            X = m[:, None] + sigma * Y.T
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            budget -= lam
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            Y = Y[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update Mean
            m_new = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Path Evolution Strategies
            Y_mu = np.sum(w[:, None] * Y[:mu], axis=0)
            pc = (1 - c_c) * pc + c_c * (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Y_mu
            
            # h_sigma condition
            norm_ps = np.linalg.norm(ps)
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * budget / lam))) < (1 + c_sigma) * np.sqrt(n + 1) / np.sqrt(1 - (1 - c_sigma)**2)
            
            # Update Covariance
            if h_sigma:
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + 1 - c_c)
            else:
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + 1 - c_c)
            
            # Rank-mu update
            rank_mu = np.sum(w[:, None]**2 * np.outer(Y[:mu], Y[:mu]).transpose(2, 0, 1), axis=2)
            C += c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update Sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * budget / lam)) - 1))
            
            m = m_new
            
            # Time Guard
            if time.time() - self.start_time > 290:
                break
                
        # Local refinement with remaining budget
        while budget > 0:
            # Simple step from best_x
            step = 0.1 * rng.standard_normal(n)
            y = np.clip(best_x + step, -5, 5)
            fy = func(y)
            budget -= 1
            if fy < best_f:
                best_x = y
                best_f = fy
                
        return best_x, best_f
