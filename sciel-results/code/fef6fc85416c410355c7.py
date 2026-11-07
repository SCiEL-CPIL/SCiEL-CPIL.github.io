"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with global-to-local strategy for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        n = self.dim
        total_budget = self.budget
        
        # Phase 1: Global exploration with random samples
        # Use 50 samples to find a good starting region
        rng = np.random.default_rng(42)
        n_global = 50
        global_calls = 0
        best_x = None
        best_f = float('inf')
        
        # Generate global samples
        global_points = rng.uniform(-5, 5, (n_global, n))
        for i in range(n_global):
            if global_calls >= total_budget or time.time() > deadline:
                break
            fx = func(global_points[i])
            global_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = global_points[i].copy()
        
        if best_x is None:
            # Fallback: single random point
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            global_calls += 1
        
        # Phase 2: Local optimization with CMA-ES from best global point
        remaining_budget = total_budget - global_calls
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
        
        # CMA-ES parameters
        # Smaller population for local refinement
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        x = best_x.copy()
        # Start with a moderate sigma for local search
        sigma = 0.5
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Covariance matrix adaptation
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls = global_calls
        best_f_local = best_f
        best_x_local = best_x.copy()
        
        sqrt_n = np.sqrt(n)
        gen = 0
        
        while calls < total_budget:
            if time.time() > deadline:
                break
            
            gen += 1
            
            # Generate population
            B = rng.standard_normal((lam, n))
            
            # Whitening matrix using eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample
            Y = x + sigma * (B @ sqrt_C.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.empty(lam)
            valid_count = 0
            for i in range(lam):
                if calls >= total_budget or time.time() > deadline:
                    break
                fy = func(Y[i])
                F[i] = fy
                calls += 1
                valid_count += 1
                if fy < best_f_local:
                    best_f_local = fy
                    best_x_local = Y[i].copy()
            
            if valid_count < 2 or calls >= total_budget or time.time() > deadline:
                break
            
            # If we didn't evaluate all points, pad with large values
            if valid_count < lam:
                F[valid_count:] = np.inf
            
            # Sort
            F_sorted_idx = np.argsort(F)
            Y_sorted = Y[F_sorted_idx]
            F_sorted = F[F_sorted_idx]
            
            # Only use valid points for update
            n_valid = valid_count
            if n_valid < mu:
                break
            
            # Update mean
            x_new = np.sum(w[:n_valid] / np.sum(w[:n_valid]) * Y_sorted[:n_valid], axis=0)
            
            # Update step size
            y_invC = (Y_sorted[:n_valid] - x) / sigma
            weighted_sum = np.sum(w[:n_valid] / np.sum(w[:n_valid]) * y_invC, axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ weighted_sum)
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            delta_h = (1 - np.exp(-2 + 5 / (n + 1)))
            if gen > 0:
                denom = np.sqrt(1 - (1 - c_sigma)**(gen / d_sigma))
            else:
                denom = 1.0
            
            if denom > 1e-12:
                norm_ps = np.linalg.norm(ps)
                threshold = (1.4 + 2 / (n + 1)) * np.trace(C) / n
                if norm_ps / denom < threshold:
                    h_pc = delta_h
                else:
                    h_pc = 0.0
            else:
                h_pc = 0.0
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            
            for i in range(min(mu, n_valid)):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            C = (C + C.T) / 2
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sqrt_n - 1))
            
            # Clip x to prevent numerical issues
            x = np.clip(x, -10, 10)
            
            # Stagnation check: if no improvement in last 20 generations, restart
            # For simplicity, we'll just continue but with a note that this could be improved
            if gen % 20 == 0:
                # Check if sigma is too small or too large
                if sigma < 1e-8 or sigma > 10:
                    # Reset sigma to a reasonable value
                    sigma = 0.5
                    C = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
        
        # Return the overall best
        if best_f_local < best_f:
            best_f = best_f_local
            best_x = best_x_local
        
        return best_x, best_f
# EVOLVE-BLOCK-END
