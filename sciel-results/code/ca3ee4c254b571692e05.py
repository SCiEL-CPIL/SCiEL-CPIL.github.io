import numpy as np
import time

class Optimizer:
    """
    Optimizer using a lightweight CMA-ES implementation.
    CMA-ES is robust to scaling and rotation, which is beneficial for 
    the hidden transformations applied to the Schwefel function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        
        # CMA-ES Parameters
        dim = self.dim
        budget = self.budget
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Initial step size
        sigma = 1.0
        # Initial mean (center of the search space)
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Step size control parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Hessian of the log determinant of C (approximated for small dim)
        # For dim=5, we can just compute det or use eigenvalues if needed, 
        # but for C update, we just need the rank-1 and rank-mu updates.
        
        # Initial evaluations
        # We need to evaluate the initial population to start the loop.
        # To save evaluations, we can start the main loop and let it handle the first gen.
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # We will use the full budget.
        # Each generation uses lambda_ evaluations.
        # Total generations = budget // lambda_
        
        # To maximize usage, we can do partial populations at the end, 
        # but standard CMA-ES works best with full populations.
        # With dim=5, lambda_ = 4 + 3*log(5) ~ 8. 
        # 500 // 8 = 62 generations. This is plenty.
        
        # If budget is not divisible, we might waste some evals at the end.
        # We can adjust lambda_ or just stop early.
        
        # Let's use a slightly smaller lambda to fit better? 
        # No, standard lambda is fine. We'll just stop when budget is reached.
        
        # To be safe and simple, we implement the core loop.
        
        # Initial point evaluation
        # Actually, CMA-ES samples around the mean. The mean is not necessarily evaluated.
        # We need to evaluate the sampled points.
        
        # Optimization loop
        evals_used = 0
        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Sample population
            # D = C^(1/2)
            # For dim=5, eigendecomposition is fast
            d, V = np.linalg.eigh(C)
            d = np.sqrt(np.maximum(d, 0))
            D = V @ np.diag(d) @ V.T
            B = V @ np.diag(1.0 / np.maximum(d, 1e-12)) @ V.T # For inverse
            
            # Generate normal vectors
            z = np.random.randn(lambda_, dim)
            # Sample
            pop = x_mean + sigma * (z @ D.T) # Note: D is symmetric, so D.T = D
            # Actually, z @ D.T is correct if z is (lambda, dim) and D is (dim, dim).
            # y = z * D -> each row is z_i * D
            
            # Evaluate population
            f_vals = np.zeros(lambda_)
            for i in range(lambda_):
                if evals_used >= budget:
                    break
                if time.time() > deadline:
                    break
                f_vals[i] = func(pop[i])
                evals_used += 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = pop[i].copy()
            
            # If we broke out of the inner loop early due to budget/time, break outer
            if evals_used >= budget or time.time() > deadline:
                break
                
            # Sort by fitness
            # Only sort the evaluated part if we didn't fill the whole pop
            # But we assume we filled it unless budget ran out.
            # If budget ran out in the middle, we can't do a proper CMA-ES update.
            # So we break if we didn't complete a generation.
            if evals_used < budget + lambda_: # This check is tricky. 
                # Let's just check if we evaluated all lambda_ points in this generation
                pass
                
            # Simpler: If we didn't evaluate all lambda_ points, break.
            # We track how many we evaluated in this gen.
            # Let's restructure to ensure we only update if we have a full generation.
            
            # Actually, the loop above increments evals_used.
            # If evals_used < budget before the loop started, we try to evaluate lambda_ points.
            # If budget < lambda_, we evaluate budget points and break.
            
            # Let's refine the logic:
            # 1. Determine how many points to evaluate in this generation (lambda_ or remaining budget)
            # 2. Evaluate them
            # 3. If we evaluated lambda_ points, do CMA-ES update.
            # 4. If not, break.
            
            # My current loop does this implicitly but the update part assumes full pop.
            # I will re-verify the code structure below.
            
            # Sort indices
            idx = np.argsort(f_vals)
            # Take best mu
            y_best = pop[idx[:mu]]
            f_best = f_vals[idx[:mu]]
            
            # Weighted mean of best points
            y_mean = np.sum(weights[:, None] * y_best, axis=0)
            
            # Update step size
            # Norm of y_mean in C^-1/2 metric
            # ||y_mean||_C^{-1} = sqrt(y_mean.T @ C^{-1} @ y_mean)
            # C^{-1} = B^2 if D is sqrt(C) and B is inv(D)? 
            # D = V diag(sqrt(d)) V^T. C = D^2. C^{-1} = V diag(1/d) V^T.
            # So C^{-1/2} = V diag(1/sqrt(d)) V^T.
            # Let's call H = C^{-1/2}
            H = V @ np.diag(1.0 / np.maximum(d, 1e-12)) @ V.T
            
            norm_y_mean = np.linalg.norm(H @ y_mean)
            
            # Expected norm of N(0, I)
            chi_lambda = np.sqrt(lambda_) * (1 - 1/(4*lambda_) + 1/(32*lambda_*lambda_))
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / norm_y_mean * y_mean if norm_y_mean > 1e-12 else (1 - c_sigma) * p_sigma
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Update p_c
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1 + 2/(dim+1)) * chi_lambda else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update C
            # Rank-1 update
            C = (1 + c_1 * h_sigma * (h_sigma - 1) - c_1) * C
            # This formula is: C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (h_sigma - 1) * C)
            # Let's use the standard update:
            C = (1 + c_1 * h_sigma * (h_sigma - 1) - c_1) * C + c_1 * (p_c @ p_c.T)
            # Wait, the standard is:
            # C <- (1 + c1 * hs * (hs - 1) - c1) * C + c1 * pc * pc^T
            # If hs=1: C <- (1 - c1) * C + c1 * pc * pc^T
            # If hs=0: C <- (1 - c1) * C
            
            # Rank-mu update
            # C <- C + c_mu * sum(weights_i * (y_i - y_mean) * (y_i - y_mean)^T)
            # Note: y_i here are the selected best points, centered by their weighted mean?
            # Standard CMA-ES uses the weighted mean of the selected points as the new mean.
            # The deviations are from this weighted mean.
            y_diffs = y_best - y_mean
            C += c_mu * np.sum(weights[:, None, None] * (y_diffs[:, :, None] * y_diffs[:, None, :]), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_mean + sigma * y_mean
            # Actually, the new mean is x_mean + sigma * y_mean?
            # In standard CMA-ES, the mean is updated by the weighted mean of the sampled points.
            # The sampled points are x_mean + sigma * z @ D.
            # So y_best are the z @ D parts.
            # New x_mean = x_mean + sigma * y_mean_weighted.
            # Yes.

        return best_x, best_f

# The above implementation is a bit messy with the partial generation handling and time checks inside the eval loop.
# Let's write a cleaner version.
