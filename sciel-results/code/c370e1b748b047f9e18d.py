"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with (1+1)-ES polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def cma_es(x0, fx0):
            nonlocal best_x, best_f, calls_made
            
            # CMA-ES parameters
            n = dim
            lambda_ = 4 + int(3 * np.log(n))  # ~7 for n=5, let's use 10 for stability
            lambda_ = 10 
            mu = lambda_ // 2  # 5
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            sigma = 1.0  # Initial step size
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            inv_C_sqrt = np.eye(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            x = x0.copy()
            f_x = fx0
            
            # Precompute Cholesky decomposition update helper (simplified)
            # We will update C using eigendecomposition every 10 generations or so? 
            # No, standard CMA-ES updates C every generation.
            # For n=5, eigendecomposition is cheap.
            
            for gen in range(1000):
                if time.time() > deadline - 5.0 or calls_made >= budget:
                    break
                
                # Sample lambda_ points
                ys = []
                xs = []
                fvals = []
                
                for i in range(lambda_):
                    if calls_made >= budget:
                        break
                    # Standard normal sample
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    
                    f_y = safe_call(y)
                    if f_y is None:
                        return
                    
                    ys.append(z)
                    xs.append(y)
                    fvals.append(f_y)
                
                if len(fvals) < lambda_:
                    break
                    
                # Sort by fitness
                order = np.argsort(fvals)
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update mean
                x_new = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
                
                # Update step size
                sigma_old = sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(C_sqrt @ np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)) / chi_n - 1))
                
                # Update covariance matrix
                # Delta h
                h_sigma = (1 - np.exp(-2 * c_sigma * (1 - c_sigma))) * np.linalg.norm(C_sqrt @ np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)) / chi_n < (1 + 2 / (n + 1)) * (1 - c_c)
                h_sigma = 1.0 if h_sigma else 0.0
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * c_c * (1 / sigma) * (x_new - x)
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (xs[i] - x) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (1 - np.trace(C) / n) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure C is positive definite (symmetrize)
                C = (C + C.T) / 2
                
                # Update C_sqrt (Cholesky)
                # Simple approach: Recompute Cholesky every generation (n=5 is fast)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Add small epsilon to diagonal if not PD
                    C += 1e-6 * np.eye(n)
                    C_sqrt = np.linalg.cholesky(C)
                    
                x = x_new
                f_x = fvals[0]
                
                # Check convergence
                if sigma < 1e-10:
                    break

        # Initialize CMA-ES
        # Use origin as start, or random? Origin is usually safe for BBOB.
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is None:
            return best_x, best_f
        
        cma_es(x0, f0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
