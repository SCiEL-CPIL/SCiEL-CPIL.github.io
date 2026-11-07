"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for the BBOB f9 task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        dim = self.dim
        # Initial population size
        popsize = 4 + int(3 * np.log(dim))
        # Initial step size (sigma)
        sigma = 2.0
        
        # Initial mean (center of the search)
        # Starting at 0 is reasonable for [-5, 5] box, though Rosenbrock min is at 1
        # The hidden shift/rotation makes the exact location unknown.
        mean = np.zeros(dim)
        
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        
        # Adaptation parameters
        # Learning rates for sigma and C
        mu = popsize // 2
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        cs = (mueff + 0.3) / (dim + mueff + 1.3)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        
        # Covariance matrix adaptation parameters
        c1 = (2 - mueff * (1 - cs**2) / (dim**2 + 1)) / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Memory for path evolution
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky decomposition of C for sampling
        # We update Cholesky periodically or use eigendecomposition if C becomes ill-conditioned
        # For simplicity and speed in Python, we will use eigendecomposition every few steps
        # or just use the standard update and re-decompose if needed.
        # Actually, for dim=5, we can just do the matrix update and use np.linalg.cholesky
        # but if it fails, we regularize.
        
        # State tracking
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        best_x = mean.copy()
        best_fx = func(best_x)
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # Get Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C if it became singular
                C += 1e-12 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
            
            # Sample standard normal vectors
            z = rng.standard_normal((popsize, dim))
            # Transform to distribution N(mean, sigma^2 * C)
            y = mean[None, :] + sigma * (z @ L.T)
            
            # Clip to box [-5, 5]
            y = np.clip(y, -5, 5)
            
            # Evaluate fitness
            fy = np.array([func(yp) for yp in y])
            calls_made += popsize
            
            # Update best found
            idx_best = np.argmin(fy)
            if fy[idx_best] < best_fx:
                best_fx = fy[idx_best]
                best_x = y[idx_best].copy()
                
            # 2. Recombine
            # Sort by fitness
            order = np.argsort(fy)
            y_sorted = y[order]
            fy_sorted = fy[order]
            
            # Weighted average of the best mu points
            mean_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # 3. Adapt covariance matrix
            # Update step size
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (mueff - 1))
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + cs * hsig * (np.linalg.norm(y_sorted[:mu] - mean) / sigma)
            # Note: The standard update uses the weighted sum of (y_i - mean) / sigma
            # Actually, the standard CMA-ES update for p_sigma is:
            # p_sigma = (1-cs)*p_sigma + cs * hsig * (1/sigma) * sum(w_i * (y_i - mean))
            # Let's correct the p_sigma update to follow the standard algorithm more closely.
            
            # Recalculate p_sigma update properly
            y_weighted = np.sum(weights[:, None] * (y_sorted[:mu] - mean), axis=0)
            p_sigma = (1 - cs) * p_sigma + cs * hsig * (y_weighted / sigma)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) - 1))
            
            # Update p_c
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (mueff - 1))
            p_c = (1 - cc) * p_c + cc * hsig * (y_weighted / sigma)
            
            # Update C
            rank_one_update = np.outer(p_c, p_c)
            rank_mu_update = np.sum(weights[:, None, None] * np.outer(y_sorted[:mu] - mean, y_sorted[:mu] - mean), axis=0) / sigma**2
            
            C = (1 + c1 * cc * (2 - cc) - c1 - cmu) * C + c1 * rank_one_update + cmu * rank_mu_update
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # If sigma becomes too small, we might be stuck.
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
