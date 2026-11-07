import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim + 1))
        if lam < 4: lam = 4
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Strategy parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/(mu_eff)) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x_mean
        best_f = float('inf')
        calls = 0
        
        # Pre-compute B? No, C changes.
        
        while calls + lam <= budget:
            if time.time() > deadline:
                break
                
            # Sample
            # Compute B = sqrt(C)
            # Use eigen decomposition for stability if Cholesky fails, but Cholesky is faster
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, project to PD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = np.linalg.cholesky(C)
                
            Y = []
            for i in range(lam):
                z = np.random.randn(dim)
                x = x_mean + sigma * (B @ z)
                Y.append(x)
            Y = np.array(Y)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lam
            
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                
            # Compute y vectors (steps)
            Y_steps = (Y[:mu] - x_mean) / sigma
            
            # Update Mean
            y_weighted = np.sum(weights[:, np.newaxis] * Y_steps, axis=0)
            x_mean = x_mean + sigma * y_weighted
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ y_weighted)
            
            # h_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((calls + 1) / ds)) < (1.4 + 2 / (dim + 1)) * chi_n)
            
            # Update p_c
            p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * y_weighted
            
            # Update C
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            
            # Rank-mu update
            # Sum of w_i * (y_i - y_weighted) * (y_i - y_weighted)^T
            # Note: y_i are the steps.
            diff = Y_steps - y_weighted
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0)
            
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu
            
            # Regularize
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update Sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
        return best_x, best_f
