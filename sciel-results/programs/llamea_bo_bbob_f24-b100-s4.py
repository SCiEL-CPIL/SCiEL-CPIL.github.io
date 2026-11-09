import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES Global/Intermediate Search ---
        # CMA-ES parameters for dimension n
        # Standard population size
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initialize CMA-ES state
        # Start at origin, which is a common heuristic, but use a larger sigma to cover the box [-5,5]
        mean = np.zeros(n)
        sigma = 2.0  # Initial step size
        C = np.eye(n)
        C_inv_sqrt = np.eye(n) # We will approximate or use identity for simplicity if full update is too slow/complex, but let's do a simplified update
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Helper to compute inverse square root of C
        # For 5D, we can use eigendecomposition periodically or just maintain C and use it for sampling
        # Sampling: y = z * C_inv_sqrt * sigma? No, x = mean + sigma * C_sqrt @ z
        # We need C_sqrt. Let's maintain C and compute C_sqrt via Cholesky each generation (cheap for 5D)
        
        def get_C_sqrt(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-10 * np.eye(n))
                return L.T # Upper triangular such that C = L.T @ L
            except:
                # Fallback if not positive definite
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        # Main CMA-ES Loop
        # Reserve ~50 calls for final polish
        cmaes_budget_limit = total_budget - 50
        
        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break
            
            # Generate population
            z = np.random.standard_normal((lambda_, n))
            C_sqrt = get_C_sqrt(C)
            
            # Sample points
            # x_i = mean + sigma * (C_sqrt @ z_i).T
            # Note: z is (lambda, n). C_sqrt is (n, n).
            # We want x_i = mean + sigma * z_i @ C_sqrt.T ?
            # Standard: y_i ~ N(0, C). x_i = mean + sigma * y_i.
            # y_i = C_sqrt.T @ z_i ? If C = L L.T, then y = L.T z has cov L.T I L = C.
            # So y_i = C_sqrt.T @ z_i.
            y = np.einsum('ij,jk->ik', z, C_sqrt.T) 
            x_pop = mean + sigma * y
            
            # Evaluate population
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])
            
            # Check if we ran out of budget during evaluation
            if calls_made >= cmaes_budget_limit:
                break
                
            # Sort indices by fitness
            idx = np.argsort(f_pop)
            
            # Update mean
            # x_w = sum(w_i * x_sorted[i])
            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mueff) * (C_inv_sqrt @ (x_w - mean)) / sigma
            # Approximation: use identity for C_inv_sqrt initially, update later if needed.
            # For simplicity and speed in 5D, we can update C and recompute inverse sqrt.
            # Let's compute C_inv_sqrt properly.
            try:
                C_inv_sqrt = np.linalg.inv(get_C_sqrt(C))
            except:
                C_inv_sqrt = np.eye(n)
                
            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3))
            # Simplified h_sigma check
            if np.linalg.norm(p_sigma) < (1.4 + 2/(n+1)) * np.sqrt((mueff+1)/(mueff+3)):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma
            
            # Update Covariance C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1-h_sigma)*c_c*(1-c_c)*C) + c_mu * sum(w_i * y_i @ y_i.T)
            # y_i are the selected normalized steps
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
            
            # Update mean
            mean = x_w
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            
            # Ensure sigma doesn't become too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Check convergence
            if sigma < 1e-8:
                break

        # --- Phase 2: Local Polish (Coordinate Descent) ---
        # Use the best point found by CMA-ES
        if best_x is not None and calls_made < total_budget:
            # Deterministic multi-scale local search
            step_sizes = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005, 0.0001]
            
            for step in step_sizes:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                        continue  # best_x updated in evaluate
                    
                    # Try negative direction
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True
                        
                if not improved and step < 0.001:
                    break

        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
