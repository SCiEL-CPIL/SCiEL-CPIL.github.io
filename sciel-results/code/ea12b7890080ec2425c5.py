import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 270.0
        
        best_x = np.zeros(n)
        best_f = float('inf')
        evals_left = budget

        # --- Phase 1: Coarse Grid Search ---
        # For n=5, a 2^5 grid costs 32 evals.
        # This establishes a good starting point and avoids the "cold start" of CMA-ES at origin if the shift is large.
        
        def generate_grid_points(low, high, num_subdivisions):
            indices = np.arange(num_subdivisions)
            grids = np.meshgrid(*([indices] * n), indexing='ij')
            grid_indices = np.stack(grids, axis=-1)
            grid_indices = grid_indices.reshape(-1, n)
            
            cell_width = (high - low) / num_subdivisions
            centers = low + (grid_indices + 0.5) * cell_width
            return centers

        low = np.full(n, -5.0)
        high = np.full(n, 5.0)
        num_subdivisions = 2
        grid_points = generate_grid_points(low, high, num_subdivisions)
        
        if evals_left < len(grid_points):
            # Fallback: just use origin if budget is tiny (unlikely given budget=500)
            f = func(np.zeros(n))
            best_f = f
            best_x = np.zeros(n)
            evals_left -= 1
        else:
            # Evaluate coarse grid
            for i in range(len(grid_points)):
                f = func(grid_points[i])
                if f < best_f:
                    best_f = f
                    best_x = grid_points[i].copy()
                evals_left -= 1
            
            # --- Phase 2: CMA-ES Refinement ---
            # Initialize CMA-ES starting from the best grid point.
            # This hybrid approach uses the grid to "warm start" the mean,
            # allowing CMA-ES to spend its budget on fine-grained adaptation
            # rather than broad exploration.
            
            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(n))
            # Start with a moderate sigma. The grid cell width is 5.0.
            # A sigma of 1.0 or 2.0 is reasonable for local refinement.
            sigma = 1.0
            x_mean = best_x.copy()
            C = np.eye(n)
            
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Main CMA-ES Loop
            while evals_left > 0:
                if time.time() > deadline:
                    break
                
                # Generate population
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                
                z = np.random.randn(lambda_, n)
                y = sigma * (z @ B.T)
                x = x_mean[None, :] + y
                
                # Evaluate fitness
                if evals_left < lambda_:
                    # Evaluate remaining points one by one
                    for i in range(evals_left):
                        f = func(x[i])
                        if f < best_f:
                            best_f = f
                            best_x = x[i].copy()
                        evals_left -= 1
                    break
                
                f = np.array([func(x[i]) for i in range(lambda_)])
                evals_left -= lambda_
                
                # Sort by fitness
                idx = np.argsort(f)
                x_sorted = x[idx]
                f_sorted = f[idx]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                
                # Recombination
                x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
                
                # Update step size
                # Compute C^{-1/2} (x_new - x_mean) / sigma
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)
                
                # Simple h_sigma check: if p_sigma norm is too large, dampen update
                # Threshold is roughly (1 + 2/(n+1)) * E||N(0,I)||
                E_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                threshold = (1 + 2/(n+1)) * E_norm
                # We use a moving average approximation or just a static check for robustness
                # A more robust check: if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2}) > threshold
                # For simplicity in this short budget, we use a heuristic:
                # If the norm of the update vector is exploding, we might want to reset or dampen.
                # However, standard CMA-ES is quite stable. Let's stick to h_sigma=1 for now
                # but add a safety check if sigma gets too big.
                h_sigma = 1.0
                
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new - x_mean) / sigma)
                
                # Update C
                y_sorted = (x_sorted[:mu] - x_mean[None, :]) / sigma
                C_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_mu
                
                # Update mean
                x_mean = x_new
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
                sigma = np.clip(sigma, 1e-12, 1e12)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals_check = np.linalg.eigvalsh(C)
                if np.any(eigvals_check < 1e-12):
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    sigma = 1.0

        return best_x, best_f
