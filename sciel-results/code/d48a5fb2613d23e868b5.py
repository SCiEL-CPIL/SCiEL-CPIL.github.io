"""CMA-ES with adaptive restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n)) for n=5 gives 4+3*1.609=4+4.8=8, round to 10
        # For dim=5, lambda=10 is reasonable. Budget 500 / 10 = 50 generations.
        pop_size = 10
        
        # Initial mean: center of box
        mean = np.zeros(dim)
        
        # Initial sigma: half the range of the box
        sigma = 5.0
        
        # Covariance matrix: identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)  # Ensure positive
        
        # Evolution path weights
        # Mu: number of best individuals used for recombination
        mu = pop_size // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # For adaptation of sigma
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # For adaptation of covariance matrix
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial best
        # Evaluate initial population
        f_vals = []
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            x = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
            x = np.clip(x, -5.0, 5.0)
            f_vals.append(call_func(x))
        
        f_vals = np.array(f_vals)
        idx_best = np.argmin(f_vals)
        
        # Store all sampled points for recombination
        all_x = np.zeros((pop_size, dim))
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            all_x[i] = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
            all_x[i] = np.clip(all_x[i], -5.0, 5.0)
        
        # Re-evaluate to get consistent values (or just use the first evaluation)
        # Actually, let's just use the f_vals from the first pass and keep the x values
        # But we need to store the x values that correspond to f_vals
        # Let me redo this more carefully
        
        # Reset and do it properly
        calls = 0
        best_f = np.inf
        best_x = None
        
        # First generation
        pop_x = np.zeros((pop_size, dim))
        pop_f = np.zeros(pop_size)
        
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            x = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
            x = np.clip(x, -5.0, 5.0)
            pop_x[i] = x
            pop_f[i] = call_func(x)
        
        idx_best = np.argmin(pop_f)
        best_x = pop_x[idx_best].copy()
        best_f = pop_f[idx_best]
        
        stagnation_count = 0
        restart_threshold = 15  # Generations without improvement before considering restart
        best_f_at_restart = best_f
        sigma_at_restart = sigma
        mean_at_restart = mean.copy()
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            
            # Update best
            if pop_f[0] < best_f - 1e-12:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Check for restart
            if stagnation_count >= restart_threshold:
                # If we've been stuck for a while, do a local restart around best
                # Reduce sigma and re-center
                sigma *= 0.5
                mean = best_x.copy()
                C = np.eye(dim)
                C_inv = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_count = 0
                best_f_at_restart = best_f
                sigma_at_restart = sigma
                mean_at_restart = mean.copy()
                
                # Generate new population
                pop_x = np.zeros((pop_size, dim))
                pop_f = np.zeros(pop_size)
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                    x = np.clip(x, -5.0, 5.0)
                    pop_x[i] = x
                    pop_f[i] = call_func(x)
                
                if calls >= budget:
                    break
                continue
            
            # Recombination: weighted mean of top mu individuals
            new_mean = np.sum(weights[:, None] * pop_x[:mu], axis=0)
            
            # Update evolution path for sigma
            # Need to compute C^(-1/2) for the path update
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            
            # Hessian correction
            hs = np.sqrt(1 - (1 - c_sigma)**2) * np.sqrt(mu_eff - 1)
            hs = max(hs, 0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            
            # Rank-one and rank-mu updates
            delta = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0
            
            C = (1 + c_1 * delta * c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * np.outer(p_c, p_c) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(pop_x[:mu] - mean, (pop_x[:mu] - mean).T), axis=0) / sigma**2
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            eigvals_new = np.maximum(eigvals_new, 1e-12)
            C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            mean = new_mean
            
            # Generate new population
            pop_x = np.zeros((pop_size, dim))
            pop_f = np.zeros(pop_size)
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                x = np.clip(x, -5.0, 5.0)
                pop_x[i] = x
                pop_f[i] = call_func(x)
            
            if calls >= budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
