import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        n = dim
        lam = 4 + int(3 * np.log(n))
        if lam < 4: lam = 4
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        cs = (mueff + 1.3) / (n + mueff + 3)
        ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Stagnation handling
        last_best_f = best_f
        stagnation_count = 0
        max_stagnation = 40
        
        while calls_made < budget and time.time() < deadline:
            # Check if we can afford a full generation
            if budget - calls_made < lam:
                break
            
            # Generate population
            Z = np.random.randn(lam, n)
            Y = Z @ C_sqrt_inv.T
            X = x_mean + sigma * Y
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                F[i] = counted_func(X[i])
            
            # Check if we completed the generation
            if calls_made < budget and time.time() < deadline:
                # Sort by fitness
                idx = np.argsort(F)
                top_mu_idx = idx[:mu]
                top_mu_X = X[top_mu_idx]
                
                # Weighted mean
                x_new_mean = np.sum(w[:, np.newaxis] * top_mu_X, axis=0)
                
                # Update sigma
                diff = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * diff
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update p_c
                p_c = (1 - cs) * p_c + cs * np.sqrt(mueff) * diff
                
                # h_sigma
                gen = calls_made // lam
                if gen > 0:
                    denom = np.sqrt(1 - (1 - c_sigma) ** (2 * gen))
                    if np.linalg.norm(p_sigma) / denom < (1.4 + 2 / (n + 1)) * chi_n:
                        h_sigma = 1.0
                    else:
                        h_sigma = 0.0
                else:
                    h_sigma = 1.0
                
                # Update C
                C = (1 - cs * (2 - cs) + h_sigma * cs * (2 - cs) - cs) * C
                C += cs * (2 - cs - h_sigma * cs * (2 - cs)) * np.outer(p_c, p_c)
                
                # Rank-mu update
                for i in range(mu):
                    y = (top_mu_X[i] - x_mean) / sigma
                    C += w[i] * (np.outer(y - p_c, y - p_c) - np.outer(p_c, p_c))
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                # Re-orthogonalize C_sqrt_inv
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update x_mean
                x_mean = x_new_mean
                
                # Stagnation check
                if best_f < last_best_f - 1e-8:
                    last_best_f = best_f
                    stagnation_count = 0
                else:
                    stagnation_count += lam
                    
                if stagnation_count > max_stagnation:
                    # Restart
                    x_mean = np.random.uniform(-5, 5, n)
                    sigma = 2.0
                    C = np.eye(n)
                    C_sqrt_inv = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    stagnation_count = 0
                    last_best_f = best_f

        return best_x, best_f
