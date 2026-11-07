"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        # CMA-ES Parameters
        # Population size lambda
        # Standard is 4 + 3*ln(n). For n=5, lambda=8.
        # Larger lambda helps with multimodality but consumes budget.
        # Let's use lambda = 20 for better exploration in early stages.
        lam = 20
        mu = lam // 2  # Number of recombination weights
        
        # Weights: log-linear weights, normalized to sum to 1
        # c_mu = sum(weights^2) / sum(weights)^2
        weights = np.log(np.arange(mu, 0, -1) + 0.5)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation (CSA) parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = (mueff / 2) * (c_1 * 2 / (dim + 1.3) - 1)
        c_mu = min(c_mu, 1 - c_1)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 1.0  # Initial step size. Box is [-5, 5].
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute invsqrt for initial C=I
        # We need to update C_sqrt and C_inv_sqrt periodically.
        # For simplicity and speed in 5D, we can recompute every few steps or use a fast update.
        # Standard CMA-ES updates C and then computes invsqrt.
        # To keep it simple and robust, we will recompute C_inv_sqrt every step if needed, 
        # but in 5D, it's cheap.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1
        
        # Helper to compute inverse square root of a symmetric positive definite matrix
        def invsqrtm(M):
            # M is symmetric
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-12, None)
            w_inv_sqrt = 1.0 / np.sqrt(w)
            return V @ np.diag(w_inv_sqrt) @ V.T

        # Initial C_sqrt and C_inv_sqrt for I
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        while evals < budget:
            # Check time
            if time.time() > deadline - 10:
                break
                
            # Sample population
            ys = []
            fs = []
            
            # We need to generate lam samples
            # z ~ N(0, I)
            Z = np.random.randn(lam, dim)
            # y = x_mean + sigma * C_sqrt @ z
            # For efficiency, precompute C_sqrt @ Z.T
            # Y = Z @ C_sqrt.T
            
            # Update C_sqrt and C_inv_sqrt if not done recently? 
            # In standard CMA-ES, we update C, then C_inv_sqrt.
            # Let's assume we update them every generation for correctness, 
            # but in 5D it's fast.
            
            # Actually, we need C_sqrt to generate samples.
            # We maintain C_sqrt.
            
            Y = Z @ C_sqrt.T
            X = np.tile(x_mean, (lam, 1)) + sigma * Y
            
            # Evaluate
            for i in range(lam):
                # Clip to box? The task says func clips.
                # But CMA-ES works better without hard clipping if possible, 
                # but here func clips. We can just pass x.
                f = func(X[i])
                fs.append(f)
                evals += 1
                if evals >= budget:
                    break
                
                if time.time() > deadline - 5:
                    break

            if len(fs) < 2:
                break
                
            fs = np.array(fs)
            X = X[:len(fs)]
            
            # Track best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort by fitness
            idx_sort = np.argsort(fs)
            X_sorted = X[idx_sort]
            fs_sorted = fs[idx_sort]
            
            # Use only top mu
            mu_actual = min(mu, len(fs_sorted))
            if mu_actual < 1:
                break
            X_mu = X_sorted[:mu_actual]
            weights_mu = weights[:mu_actual]
            
            # New mean
            x_new = np.sum(weights_mu[:, None] * X_mu, axis=0)
            
            # Evolution path for sigma
            # h_sigma condition
            # Check if p_sigma is too long
            # ||p_sigma|| / sqrt(1 - (1-sigma)^{2/gen})
            
            # First, update p_sigma with the old C_inv_sqrt? 
            # Standard CMA-ES uses the inverse square root of the OLD covariance.
            
            # Compute p_sigma update
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            
            # We need C_inv_sqrt corresponding to the OLD C.
            # We have C_inv_sqrt from the previous iteration.
            
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(dim+1)) * sqrt((1 - (1-c_sigma)^{1/gen}) * (1 - (1-c_sigma)^{2/gen}))
            # Approximation for gen:
            gen = 1.0 # We don't track gen explicitly, but we can estimate
            # Let's use a simpler heuristic or standard formula with a fixed gen estimate
            # For robustness, let's use the standard check.
            # We need to estimate 'gen' or use a fixed value.
            # Let's track generations.
            
            # Actually, let's simplify. In 5D, CMA-ES is very fast.
            # We can recompute C_inv_sqrt every step.
            
            # Update p_sigma
            # Note: C_inv_sqrt is for the OLD C.
            delta_sigma = C_inv_sqrt @ (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_sigma
            
            # h_sigma check
            # We need an estimate of generations since last update or just use a constant.
            # Let's use a fixed threshold based on dim.
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^2) < (1.4 + 2/(dim+1))
            # This is a simplification.
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Threshold
            thresh_sigma = (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)
            h_sigma = 1.0 if norm_p_sigma < thresh_sigma else 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mueff) - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma
            
            # Rank-one update
            # C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C 
            #   + c_mu * sum(weights * (y_i - x_mean) * (y_i - x_mean).T) / sigma^2
            #   + (1 + c_c * (2 - c_c) - c_1) * p_c * p_c.T
            
            # Wait, the standard formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C)
            #   + c_mu * sum(weights_i * y_i' y_i'^T)
            # where y_i' = (y_i - x_mean) / sigma
            
            # Let's use the standard formulation:
            # Delta C = c_1 * (p_c p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C)
            #          + c_mu * sum(weights_i * y_i' y_i'^T)
            # C = (1 - c_1 - c_mu) * C + Delta C
            
            # Compute y_i'
            Y_prime = (X_mu - x_mean) / sigma
            
            # Sum of outer products
            # S = sum(weights_i * y_i' y_i'^T)
            # S = (Y_prime.T * weights_mu) @ Y_prime
            
            S = (Y_prime.T * weights_mu) @ Y_prime
            
            Delta_C = c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * S
            
            C = (1 - c_1 - c_mu) * C + Delta_C
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update C_sqrt and C_inv_sqrt
            # This is the expensive part, but in 5D it's trivial.
            # C_sqrt = sqrt(C)
            # C_inv_sqrt = invsqrt(C)
            
            # Compute eigenvalues and vectors
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-12, None)
            C_sqrt = V @ np.diag(np.sqrt(w)) @ V.T
            C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
            
            # Update mean
            x_mean = x_new
            
            # Safety check for NaN/Inf
            if not np.all(np.isfinite(x_mean)) or not np.all(np.isfinite(C)):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
