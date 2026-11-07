"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, lambda=12 is a good balance between exploration and exploitation
        lam = 12
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size (sigma)
        # Initial sigma should be reasonable relative to the box size (10)
        # Box is [-5, 5], so width is 10. A sigma of 1.0-2.0 is typical.
        sigma = 1.0
        
        # Initial center
        # Start at 0, which is the center of the box. 
        # The problem states the minimizer is in [-4, 4]^5.
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Time constants
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial evaluation
        # Clip to box
        x_mean = np.clip(x_mean, -5, 5)
        f_values = np.full(lam, np.inf)
        
        # We need to evaluate the initial population or just start the loop?
        # Standard CMA-ES starts with a population. 
        # However, we have a budget constraint. 
        # Let's generate the first population.
        
        # To be safe with the budget, we count every func call.
        # Budget is 500.
        # Lam=12. 500 // 12 = 41 generations.
        # This is quite few for CMA-ES to converge fully, but better than (1+1) for anisotropy.
        
        # Alternative: Use a smaller lambda? 
        # If lambda=4, mu=2. 
        # 500 // 4 = 125 generations. 
        # CMA-ES with small lambda is essentially a random walk with covariance adaptation.
        # Let's try lambda=6, mu=3.
        # 500 // 6 = 83 generations.
        
        # Let's stick with lambda=6 for better sample efficiency in low budget.
        lam = 6
        mu = 3
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Pre-compute matrix square root of C for sampling
        # We will update C every generation, so we need to recompute sqrt(C) or use Cholesky
        # For dim=5, Cholesky is cheap.
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Initialize population
        # We will generate the population inside the loop to manage budget strictly.
        
        while calls_made < budget and (time.time() < deadline):
            # Generate population
            # Sample from N(x_mean, sigma^2 * C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                L = np.linalg.cholesky(C)
                
            # Generate lam samples
            # y_i ~ N(0, I)
            Y = np.random.randn(lam, dim)
            # Transform to x_i
            # x_i = x_mean + sigma * (L @ y_i.T).T
            X = x_mean + sigma * (Y @ L.T)
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                calls_made += 1
                
                # Update best
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # If we ran out of budget or time in the middle of the population,
            # we might have incomplete F. We should probably break if incomplete.
            # But CMA-ES needs a full population to update.
            # If we don't have a full population, we cannot update the covariance.
            # We will just stop.
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            # Normalized step
            # C^-0.5 (x_new_mean - x_mean) / sigma
            # We have L such that C = L @ L.T. So C^-0.5 is not L^-1.
            # We need the inverse of the Cholesky factor for the normalization?
            # Actually, the standard update uses C^{-1/2}.
            # Since C = L L^T, C^{-1/2} is not simply L^{-1}.
            # However, for the path update, we usually compute:
            # y = (x_new_mean - x_mean) / sigma
            # Then we need to project y onto the "unit sphere" in the metric defined by C.
            # This requires solving C v = y, i.e., v = C^{-1} y.
            # Wait, the formula is p_sigma = (1-c_sigma)p_sigma + c_sigma * sqrt(mu_eff) * C^{-1/2} * (x_new_mean - x_mean)/sigma
            # To compute C^{-1/2} * y, we can solve C z = y, then z = C^{-1} y.
            # No, C^{-1/2} y is the vector v such that C v = y? No.
            # If C = A A^T, then C^{-1/2} = (A^T)^{-1} A^{-1}? No.
            # Let's use the property: if we have Cholesky L (lower triangular), C = L L^T.
            # We want to compute C^{-1/2} y.
            # We can compute w = L^{-T} y (solve L^T w = y), then v = L^{-1} w (solve L v = w).
            # Then v = C^{-1/2} y.
            
            y_step = (x_new_mean - x_mean) / sigma
            
            # Solve L^T w = y_step
            w = np.linalg.solve(L.T, y_step)
            # Solve L v = w
            v = np.linalg.solve(L, w)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * v
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))
            # Correction: The h_sigma formula depends on the number of generations.
            # gen = calls_made // lam
            # h_sigma = (norm(p_sigma) / sqrt(1 - (1-c_sigma)^(2*gen))) / sqrt(1 - (1-c_sigma)^(2*gen)) ? 
            # Standard formula: h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))) / sqrt(1 - (1-c_sigma)^(2*gen)) is wrong.
            # It is: h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))) / E[||N(0, I)||]
            # E[||N(0, I)||] = sqrt(dim) * Gamma((dim+1)/2) / (sqrt(pi) * Gamma(dim/2))
            # For dim=5, this is a constant. Let's approximate or precompute.
            # For dim=5, E[||N(0,I)||] is approx 2.355 (actually sqrt(5) * Gamma(3) / (sqrt(pi) Gamma(2.5)) = 2.236 * 2 / (1.772 * 1.329) = 4.472 / 2.355 = 1.898? No.
            # Let's just use the standard constant for dim=5.
            # c_n = sqrt(dim) * Gamma((dim+1)/2) / (sqrt(pi) * Gamma(dim/2))
            # For dim=5: sqrt(5) * Gamma(3) / (sqrt(pi) * Gamma(2.5))
            # Gamma(3) = 2
            # Gamma(2.5) = 1.5 * 0.5 * sqrt(pi) = 0.75 * sqrt(pi)
            # c_n = sqrt(5) * 2 / (sqrt(pi) * 0.75 * sqrt(pi)) = 2 * sqrt(5) / (0.75 * pi) = 2.236 / 2.356 = 0.949?
            # Wait, Gamma(2.5) = 1.5 * 0.5 * sqrt(pi) = 0.75 * 1.772 = 1.329.
            # c_n = 2.236 * 2 / (1.772 * 1.329) = 4.472 / 2.355 = 1.899.
            
            c_n = 1.899 # Approx for dim=5
            
            gen = calls_made // lam
            # If gen is 0, the term is 1.
            if gen == 0:
                h_sigma = 1.0
            else:
                norm_p_sigma = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
                h_sigma = norm_p_sigma / denom / c_n
                
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Update covariance
            # delta_h = (1 + (1 - h_sigma) * c_c * (2 - c_c)) * (1 - c_1) - 1 ? No.
            # The correction term for the covariance update when h_sigma != 1.
            # delta_h = (1 + (1 - h_sigma) * c_c * (2 - c_c)) * (1 - c_1) - 1
            delta_h = (1 + (1 - h_sigma) * c_c * (2 - c_c)) * (1 - c_1) - 1
            
            # Rank-1 update
            # p_c p_c^T
            # Rank-mu update
            # sum weights (x_i - x_mean_old) (x_i - x_mean_old)^T / sigma^2
            
            # Note: x_i are the selected ones.
            # We use the OLD mean for the difference.
            diff = (X_sorted[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            
            C = (1 + c_1 * (delta_h - 1) - c_mu) * C + c_1 * (np.outer(p_c, p_c) - delta_h * C) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean and sigma
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / c_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check convergence
            if np.linalg.norm(x_new_mean - x_mean) < 1e-6:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
