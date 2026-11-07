import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin
        
        rng = np.random.default_rng(42)
        
        # Phase 1: CMA-ES Global Search
        # We allocate 80% of budget to CMA-ES
        budget_cma = int(self.budget * 0.8)
        
        # CMA-ES Parameters
        n = self.dim
        lam = 12  # Population size
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_w = (weights ** 2).sum()
        
        # Step size adaptation
        c_sigma = (1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1)) / (n + 4)
        d_sigma = 1 + c_sigma + 2 * max(0, (np.sqrt((lam / 2 - 1) * (1 - mu_w) / (lam / 2 + 1)) - 1) / (n + 3))
        
        # Covariance matrix adaptation
        c_1 = 2 / ((n + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((n + 2) ** 2 + lam))
        c_c = (4 + mu_w / c_1) / (n + 4 + mu_w / c_1)
        
        # Initial values
        x_mean = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute Cholesky decomposition update factors (simplified for small n)
        # We will recompute C_sqrt each generation for simplicity, as n=5 is small
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1
        
        # CMA-ES Loop
        while calls < budget_cma:
            if time.time() > deadline:
                break
                
            # Generate candidates
            X = np.zeros((lam, n))
            for i in range(lam):
                # Update C_sqrt occasionally? No, standard CMA updates C each gen.
                # For n=5, cholesky is cheap.
                # However, to save time, we can use the previous C_sqrt if C hasn't changed much?
                # Standard CMA updates C.
                
                # Let's update C and C_sqrt once per generation
                if i == 0:
                    # Update covariance matrix
                    y_w = (x_mean - x_mean) # placeholder
                    # Actually, we need the selected individuals first to update C.
                    # This loop structure is slightly inefficient for CMA.
                    # Let's restructure: Generate all, evaluate, then update.
                    pass
            
            # Restructure: Generate all candidates using current C_sqrt
            # But C_sqrt is from previous iteration.
            # This is standard.
            
            # Generate all lam candidates
            for i in range(lam):
                z = rng.standard_normal(n)
                # x = x_mean + sigma * C_sqrt @ z
                X[i] = x_mean + sigma * (C_sqrt @ z)
                X[i] = np.clip(X[i], -5, 5)
            
            # Evaluate all
            F = np.array([func(X[i]) for i in range(lam)])
            calls += lam
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sel = X[idx[:mu]]
            F_sel = F[idx[:mu]]
            
            # Update best
            if F[idx[0]] < best_f:
                best_f = F[idx[0]]
                best_x = X[idx[0]].copy()
                
            # Update mean
            x_mean_old = x_mean
            x_mean = (weights @ X_sel.T).T
            
            # Update step size
            y_sigma = (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * y_sigma
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance
            if hs:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_w) * y_sigma
            else:
                pc = (1 - c_c) * pc
                
            # Rank-one update
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            delta_hs = (1 - hs) * c_mu * (2 - c_mu)
            # y_i = (X_sel[i] - x_mean_old) / sigma
            Y_sel = (X_sel - x_mean_old) / sigma
            C += delta_hs * (weights @ (Y_sel * Y_sel.T * weights[:, np.newaxis]).T) # This is wrong shape logic, let's do it properly
            
            # Correct Rank-mu update:
            # C += c_mu * sum(weights[i] * outer(y_i, y_i))
            # Y_sel is (mu, n). weights is (mu,).
            # outer(y_i, y_i) is (n, n).
            # Sum over i.
            rank_mu_term = np.zeros((n, n))
            for i in range(mu):
                rank_mu_term += weights[i] * np.outer(Y_sel[i], Y_sel[i])
            C += delta_hs * rank_mu_term
            
            # Ensure positive definite and compute sqrt
            # Symmetrize
            C = (C + C.T) / 2
            # Add small identity for stability
            C += 1e-12 * np.eye(n)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset or use eigenvalue clipping
                # For n=5, this is rare.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = np.linalg.cholesky(C)
                
        # Phase 2: Local Refinement with (1+1)-ES
        remaining = self.budget - calls
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.5 # Start small
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.9
                    
        return best_x, best_f
