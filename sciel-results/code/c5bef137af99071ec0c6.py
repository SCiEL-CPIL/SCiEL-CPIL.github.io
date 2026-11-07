import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        lambda_ = 8
        mu = lambda_ // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        cs = (mu_eff + 2) / (N + mu_eff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + cs
        
        x_mean = np.zeros(N)
        sigma = 1.0
        C = np.eye(N)
        
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        rng = np.random.default_rng(0)
        
        def cholesky_with_fallback(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return np.linalg.cholesky(C + 1e-8 * np.eye(N))
        
        B = cholesky_with_fallback(C)
        B_inv = np.linalg.inv(B)
        
        evaluations = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        evaluations += 1
        
        while evaluations + lambda_ <= self.budget:
            Y = rng.standard_normal((lambda_, N))
            X = x_mean + sigma * (Y @ B.T)
            
            F = np.array([func(x) for x in X])
            evaluations += lambda_
            
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
            
            x_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            delta = (x_new - x_mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B_inv @ delta)
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            # Calculate h_sigma
            # Threshold: (1.4 + 2/(N+1)) * sqrt(1 - (1-cs)^(2*(evaluations+1)))
            # Note: evaluations is the count AFTER this generation
            threshold = (1.4 + 2 / (N + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (evaluations + 1)))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
            
            # Update p_c
            p_c = (1 - cc) * p_c + h_sigma * cc * delta
            
            # Update C
            Y_mu = (X_sorted[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, np.newaxis] * np.einsum('ni,nj->nij', Y_mu, Y_mu), axis=0)
            
            # Standard CMA-ES update
            # C = (1 + c1*(2*h_sigma - 1) - c1) * C + c1 * (p_c p_c^T - (1-h_sigma)C) + cmu * rank_mu
            # Simplified:
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (2*h_sigma - 1) * C - (1-h_sigma)C) ... 
            # Let's use the explicit formula:
            C = (1 + c1 * (2 * h_sigma - 1) - c1) * C
            C += c1 * (np.outer(p_c, p_c) - (1 - h_sigma) * C)
            C += cmu * rank_mu
            
            C = (C + C.T) / 2
            
            # Update Sigma
            sigma_n = np.sqrt(N)
            sigma *= np.exp((cs / damps) * (norm_p_sigma / sigma_n - 1))
            
            # Update Mean
            x_mean = x_new
            
            # Update Cholesky
            B = cholesky_with_fallback(C)
            B_inv = np.linalg.inv(B)
            
            if sigma < 1e-12:
                break

        return best_x, best_f
