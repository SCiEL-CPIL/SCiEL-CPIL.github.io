import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0 # Safety margin
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sampling
        n_init = 10
        for _ in range(n_init):
            x = rng.uniform(-5.0, 5.0, n)
            try_func(x)

        # Phase 2: CMA-ES
        # Parameters
        lambda_ = 8
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        mean = best_x.copy()
        sigma = 2.0
        
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Precompute B and D for C = B D^2 B^T
        B = np.eye(n)
        D = np.ones(n)
        
        while calls < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                # x = mean + sigma * B @ (D * z)
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5.0, 5.0)
                f = try_func(x)
                offspring.append((f, x))
                
            if not offspring:
                break
                
            # Sort
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            
            # Update ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D @ (new_mean - mean)))
            # Note: (new_mean - mean) is in the original space.
            # The update formula usually involves the inverse of the transformation.
            # Standard CMA-ES:
            # y_mean = (new_mean - mean) / sigma
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * B^T @ (C^-1/2 @ y_mean)
            # Since we maintain C = B D^2 B^T, C^-1/2 = B D^-1 B^T.
            # So C^-1/2 @ y_mean = B D^-1 B^T y_mean.
            # Then B^T @ (B D^-1 B^T y_mean) = D^-1 B^T y_mean.
            # This is getting complicated to implement from scratch correctly without bugs.
            
            # Alternative: Use a simpler adaptation or just rely on the mean update and sigma adaptation with a fixed or slowly adapting covariance?
            # Or, implement the standard update using matrix operations.
            
            # Let's use the standard update rules.
            # We need C^-1/2. We can update B and D via eigenvalue decomposition of C each step, or use the rank-1/rank-mu updates.
            # For n=5, eigenvalue decomposition is cheap.
            
            # 1. Update ps
            # y_mean = (new_mean - mean) / sigma
            y_mean = (new_mean - mean) / sigma
            
            # Compute C^-1/2
            # C = B @ np.diag(D**2) @ B.T
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ y_mean)
            
            # 2. Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            
            # 3. Update C
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (||pc||^2 / n - 1) * I) + cmu * sum(weights * (xi - mean)(xi - mean).T / sigma^2)
            
            delta_sig = (np.linalg.norm(ps) / chi_n - 1) * cs * (2 - cs)
            
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (np.dot(pc, pc) / n - 1) * np.eye(n))
            
            for i in range(mu):
                diff = (offspring[i][1] - mean) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)
                
            # Re-orthonormalize B and D from C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # 4. Update sigma
            sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
            
            # 5. Update mean
            mean = new_mean
            
            # Safety check for NaN
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sigma)) or sigma < 1e-10:
                break

        return best_x, best_f
