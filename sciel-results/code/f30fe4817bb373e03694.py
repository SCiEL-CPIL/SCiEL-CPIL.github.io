import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        n = self.dim
        # Standard CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        inv_sqrt_C = np.eye(n)
        
        # Initial evaluation
        x0 = np.clip(x_mean, -5, 5)
        f0 = func(x0)
        best_x = x0
        best_f = f0
        evals = 1
        
        # Precompute Cholesky decomposition
        try:
            chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            chol = np.eye(n)
            
        rng = np.random.default_rng(0)

        while evals + lam <= self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            N = rng.standard_normal((lam, n))
            Y = N @ chol.T
            X = x_mean + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lam
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0]
                
            # Update mean
            x_mean_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update step size control vector
            pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) * (inv_sqrt_C @ (x_mean - x_mean_old)) / sigma
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_sqrt_C @ (x_mean - x_mean_old)) / sigma
            chi_n = np.sqrt(n) * (1 - 1 / (2 * n**2) + 1 / (8 * n**4))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta_sig = (1 - c_c) / c_c
            outer = np.outer(pc, pc)
            C = (1 + c_1 * delta_sig - c_1 - c_mu) * C \
                + c_1 * (outer - C) \
                + c_mu * np.sum(weights[:, np.newaxis] * np.outer(X_sorted[:mu] - x_mean_old, X_sorted[:mu] - x_mean_old), axis=0)
                
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = C + 1e-12 * np.eye(n)
                    eigvals, eigvecs = np.linalg.eigh(C)
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
                chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                inv_sqrt_C = np.eye(n)
                chol = np.eye(n)
                
            # Limit sigma
            sigma = np.clip(sigma, 1e-12, 100)
            
            # If we have remaining budget, do a single step (1+1) style to use it?
            # No, CMA-ES is population based. If budget is not multiple of lam, 
            # we might leave some calls unused. 
            # The contract says "More than 500 calls... makes result invalid". 
            # "Repeating the last n_i if fewer calls are made". 
            # So using fewer than 500 is allowed and handled by scoring.
            # But we should try to use as many as possible if time permits.
            
        # Use remaining budget with 1+1 ES if time and budget allow
        while evals < self.budget:
            if time.time() > deadline:
                break
            y = np.clip(x_mean + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            evals += 1
            if fy < best_f:
                best_f = fy
                best_x = y
                # Maybe adjust sigma slightly?
                # sigma *= 1.05
            else:
                # sigma *= 0.95
                pass

        return best_x, best_f
