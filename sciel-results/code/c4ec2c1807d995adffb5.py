"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        bounds = (-5.0, 5.0)
        
        # CMA-ES parameters
        # Population size lambda
        lam = int(4 + 3 * np.log(n))
        
        # Initial sigma (step size)
        # Start with a reasonable spread, e.g., 1/3 of the range
        sigma = 1.0 
        
        # Initial mean (center of the search)
        # Start at 0, which is often near the center of the box
        mean = np.zeros(n)
        
        # Covariance matrix (initially identity)
        C = np.eye(n)
        
        # Inverse covariance matrix
        C_inv = np.eye(n)
        
        # Eigenvalue decomposition of C (initially trivial)
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
        
        # Evolution path for covariance adaptation
        p_c = np.zeros(n)
        # Evolution path for sigma adaptation
        p_sigma = np.zeros(n)
        
        # Learning rates
        c_sigma = (1 + 2 * max(0, np.sqrt((lam - 2) / n) - 1)) / (n + 4)
        c_c = (4 + (lam - 2) / n) / (n + 4 + 2 * (lam - 2) / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 / ((n + 2) ** 2 + lam))
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Call counter and best solution tracking
        call_count = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Pre-allocate memory for population
        # We will generate points, evaluate, and select
        
        # To strictly respect the budget, we will loop until budget is exhausted.
        # Each iteration uses 'lam' evaluations.
        
        while call_count < budget:
            # Check if we can fit another full population
            # If not, we might do a partial population or just stop.
            # For simplicity and robustness, we stop if remaining budget < lam.
            # However, to maximize usage, we can adjust lam or just accept the loss.
            # Given budget=500 and lam ~ 11-13, we can do many iterations.
            
            if budget - call_count < lam:
                # Do a final partial evaluation or just break.
                # Let's just break to avoid complexity, as CMA-ES converges well.
                break

            # Generate offspring
            # Update eigen decomposition if necessary (every 1/chi_c iterations or so)
            # For simplicity in this short script, we update every iteration or use a cheap approximation.
            # Standard CMA-ES updates eigenvalues every 1/chi_c steps.
            # Let's just do a full eigendecomposition every few steps to save time, 
            # or just every step since n=5 is small.
            
            # 1. Update evolution paths
            # We need the previous population's sorted indices to update paths.
            # This requires storing the previous population.
            
            # Let's restructure: 
            # 1. Generate population
            # 2. Evaluate
            # 3. Sort
            # 4. Update mean
            # 5. Update paths and C, sigma
            # 6. Update eigenvalues
            
            # Generate N(0, C) samples
            # We need the square root of C. We can use the eigen decomposition: C = V * D * V^T
            # So C^{1/2} = V * D^{1/2} * V^T
            # And C^{-1/2} = V * D^{-1/2} * V^T
            
            # Initial state: C=I, so C^{1/2}=I, C^{-1/2}=I
            # We will maintain C_sqrt_inv and C_sqrt
            
            # To avoid maintaining C_sqrt explicitly, we use the eigenvalues/vecs.
            # Let's re-initialize the state properly for the loop.
            
            # Actually, implementing full CMA-ES correctly in a few lines is tricky.
            # Let's use a simpler, robust variant: (1, lambda)-CMA or just a well-tuned Nelder-Mead with random restarts?
            # No, the directive says "design a new program... from an approach family... none of the current programs use".
            # Current is L-BFGS-B. CMA-ES is a distinct family (Evolution Strategy).
            
            # Let's implement a standard CMA-ES carefully.
            
            # State variables
            x_mean = np.zeros(n)
            sigma = 1.0
            C = np.eye(n)
            C_inv = np.eye(n)
            p_c = np.zeros(n)
            p_sigma = np.zeros(n)
            
            # Weights
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)
            
            # Learning rates
            c_sigma = (1 + 2 * max(0, np.sqrt((lam - 2) / n) - 1)) / (n + 4)
            c_c = (4 + (lam - 2) / n) / (n + 4 + 2 * (lam - 2) / n)
            c_1 = 2 / ((n + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 / ((n + 2) ** 2 + lam))
            
            # Chi_n is the mean of the norm of a standard normal vector
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # We will iterate
            # To save code, I will assume the user has enough budget for several generations.
            
            # Helper to get C_inv_sqrt and C_sqrt from eigen decomposition
            def get_c_sqrt(cov, eigvals, eigvecs):
                # cov = eigvecs @ diag(eigvals) @ eigvecs.T
                # sqrt(cov) = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # We need to maintain eigvals and eigvecs.
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            
            # Main Loop
            while call_count < budget:
                if budget - call_count < lam:
                    # If we can't do a full generation, we might do a single point evaluation
                    # using the current mean and sigma, but CMA-ES relies on the population.
                    # Let's just break.
                    break
                
                # 1. Sample population
                # z ~ N(0, I)
                Z = np.random.randn(lam, n)
                
                # Transform to x = mean + sigma * C^{1/2} z
                # C^{1/2} = V diag(sqrt(D)) V^T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                X = np.array([x_mean + sigma * (C_sqrt @ z.T).T for z in Z])
                
                # Clip to bounds? The problem says func clips. 
                # It is better to let the optimizer work in the unbounded space 
                # and let the func clip, OR reflect. 
                # The contract says "func(x) clips... to the box". 
                # So we can generate outside and let it clip. 
                # However, CMA-ES adapts C based on the differences. 
                # If we clip, the differences are distorted. 
                # Standard practice for bounded CMA-ES is to use a transformation 
                # or just let it clip if the bounds are not too tight. 
                # Here [-5,5] is the domain. 
                # We will generate in R^n and pass to func.
                
                # 2. Evaluate
                F = np.array([func(x) for x in X])
                call_count += lam
                
                # Update best
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = X[min_idx].copy()
                
                # 3. Sort
                sorted_indices = np.argsort(F)
                X_sorted = X[sorted_indices]
                F_sorted = F[sorted_indices]
                
                # 4. Update Mean
                x_new_mean = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
                
                # 5. Update Evolution Paths
                # p_sigma = (1-c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * C^{-1/2} * (x_new_mean - x_mean) / sigma
                # We need C^{-1/2}
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                y = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ y)
                
                # h_sigma condition
                norm_p_sigma = np.linalg.norm(p_sigma)
                E_norm_p_sigma = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count // lam + 1)))
                h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (n + 1)) * E_norm_p_sigma else 0
                
                # p_c = (1-c_c) p_c + h_sigma * c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * y
                
                # 6. Update Covariance Matrix
                # Delta H = (c_1 + c_mu) (p_c p_c^T - C)
                # + c_mu / sigma^2 sum w_i (x_i - x_mean)(x_i - x_mean)^T - C
                # Wait, the formula is:
                # C <- (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (1 - c_c) C) 
                # + c_mu sum w_i (x_i - x_mean)(x_i - x_mean)^T / sigma^2
                
                # Note: The term (x_i - x_mean) should be the difference from the OLD mean or NEW mean?
                # Standard CMA-ES uses the difference from the OLD mean for the rank-mu update part?
                # Actually, the standard update is:
                # C_{t+1} = (1 - c_1 - c_mu) C_t + c_1 (p_c p_c^T + (1-h_sigma)c_c(1-c_c)C_t) 
                #            + c_mu sum_i w_i (y_i y_i^T)
                # where y_i = (x_i - x_mean_old) / sigma
                
                Y = (X_sorted[:mu] - x_mean) / sigma
                RankMuUpdate = np.sum(w[:, np.newaxis, np.newaxis] * Y[:, :, np.newaxis] * Y[:, np.newaxis, :], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * RankMuUpdate
                
                # Symmetrize C to avoid numerical drift
                C = (C + C.T) / 2
                
                # 7. Update Sigma
                # sigma <- sigma * exp(c_sigma / chi_n * (||p_sigma|| / chi_n - 1))
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                
                # 8. Update Eigenvalues and Vectors
                # We need to update eigvals and eigvecs of C.
                # We can do a full eigendecomposition every iteration since n=5 is small.
                eigvals, eigvecs = np.linalg.eigh(C)
                
                # Ensure positive eigenvalues (numerical stability)
                eigvals = np.maximum(eigvals, 1e-12)
                
                # Update x_mean
                x_mean = x_new_mean
                
                # Safety: if sigma becomes too small or too large, stop or adjust
                if sigma < 1e-15 or sigma > 1e15:
                    break

            # After the loop, ensure we have a valid best_x
            if best_f == np.inf:
                # Fallback: evaluate 0
                best_x = np.zeros(n)
                best_f = func(best_x)
                call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
