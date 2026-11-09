"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # Leave 20s margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d)) 
        if lam < 10: lam = 10 # Ensure reasonable population for d=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Initialization
        # Start near center, but spread out
        rng = np.random.default_rng(42)
        x_mean = np.zeros(d)
        sigma = 2.0 # Initial step size, box is [-5, 5]
        
        # Covariance matrix C
        C = np.eye(d)
        # Inverse square root of C
        C_inv_sqrt = np.eye(d)
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Update parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mueff / (d + 1) - 1)) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = (2 - mueff / (d + 1) + 2 * mueff / (d + 1)) / ((d + 2) ** 2 + mueff)
        c_mu = 2 * (mueff / (d + 1) - 1) / ((d + 2) ** 2 + mueff)
        
        # Precompute
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        best_x = None
        best_f = float('inf')
        
        # Initial population
        # Generate initial population
        # We have budget calls. 
        # Each generation costs lam calls.
        # We also need to evaluate the mean if it's better? No, usually we just use the sampled points.
        # However, if the mean is inside the box, it's a good candidate.
        
        calls_made = 0
        
        # Helper to evaluate
        def eval_point(x):
            nonlocal calls_made
            if calls_made >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            return f_val

        # Initial evaluation of mean? 
        # Let's just start the loop.
        
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(d)
                L = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                sigma = 1.0
                
            # Sample
            y = rng.standard_normal((lam, d))
            # Apply C_inv_sqrt (which is L.T for standard normal to C-distributed)
            # Actually, standard CMA-ES samples z ~ N(0, I), then y = L z
            # x = x_mean + sigma * L z
            pop = x_mean[None, :] + sigma * (y @ L.T)
            
            # Clip to box
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            f_vals = np.array([eval_point(x) for x in pop])
            
            # Sort
            order = np.argsort(f_vals)
            pop_sorted = pop[order]
            f_sorted = f_vals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # If we ran out of budget during evaluation, break
            if calls_made >= budget:
                break
            
            # Update mean
            # Weighted mean of best mu individuals
            new_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # Calculate step
            # p_c update
            y_sigma = (new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam + 1)))) < (1.4 + 2 / (d + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update C
            # Rank-1 update
            rank1 = np.outer(p_c, p_c)
            # Rank-mu update
            v = pop_sorted[:mu] - x_mean
            rankmu = np.sum(weights[:, None, None] * (v[:, :, None] * v[:, None, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 - h_sigma) * c_1 * C) + c_mu * rankmu
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Recalculate Cholesky if needed (done at start of next loop)
            # We can update C_inv_sqrt if we store it, but recalculating is fine for d=5
            
            # If convergence is very fast, we might stop early? 
            # But we want to use the budget to polish.
            # CMA-ES handles this by shrinking sigma.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
