import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        D = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For D=5, lambda=6 is a good balance
        lam = 6
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cs = (mueff + 2) / (D + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        c1 = (2 - mueff / (D + 1) + cs) / ((D + 1.3)**2 + mueff + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2)**2 + mueff))
        
        # Initial State
        # Start at 0,0,0,0,0
        xmean = np.zeros(D)
        sigma = 2.0
        C = np.eye(D)
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        
        rng = np.random.default_rng(0)
        
        x_best = xmean.copy()
        f_best = func(x_best)
        calls_made = 1
        
        chiN = np.sqrt(D) * (1 - 1.0 / (4.0 * D**2) + 1.0 / (21.0 * D**4))
        
        while calls_made + lam <= budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.standard_normal((lam, D))
            X = xmean + sigma * (C_sqrt @ z.T).T
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < f_best:
                f_best = F[0]
                x_best = X[0].copy()
                
            # Update mean
            xnew = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (xnew - xmean) / sigma
            
            # h_sigma check
            # Use simplified condition
            if np.linalg.norm(p_sigma) < 1.4 * chiN:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            # Update p_c
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * C_inv_sqrt @ (xnew - xmean) / sigma
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            
            # Update C
            Y = (X[:mu] - xmean) / sigma
            rank_mu_update = np.sum(w[:, None, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            
            rank_1_update = p_c[:, None] * p_c[None, :] + (1 - h_sigma) * cc * (2 - cc) * C
            
            factor = 1 + c1 * h_sigma * (2 - cc) - c1 - cmu
            C = factor * C + c1 * rank_1_update + cmu * rank_mu_update
            
            # Symmetrize
            C = (C + C.T) / 2
            
            xmean = xnew
            
        return x_best, f_best
