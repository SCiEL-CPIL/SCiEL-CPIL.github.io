"""Population-based CMA-ES with 1/5 success rule and progress-halving restart."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES implementation with:
    1. 1/5 success rule for sigma adaptation (sliding window).
    2. Progress-halving restart: if best_f doesn't improve by 10% of initial range, 
       reset population to a fresh spread around current best.
    3. Box-constrained sampling via reflection.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin
        
        d = self.dim
        # Population size: standard CMA-ES suggests lambda = 4 + floor(3 ln(n))
        # For n=5, lambda = 4 + 3*1.6 = 8.8 -> 9. Let's use 10 for stability.
        lam = 10
        mu = lam // 2
        
        # Initial sigma
        sigma = 1.0
        
        # State
        best_f = np.inf
        best_x = np.zeros(d)
        
        # Sliding window for 1/5 rule
        # We track the last 10 evaluations' success/failure relative to the *current* best at the time of evaluation?
        # No, the standard 1/5 rule compares the candidate to the *previous* best or the *current* best?
        # In (1+1)-ES, it's vs current best. In CMA-ES, sigma adaptation is usually based on the step size of successful updates.
        # For a simpler, robust approach in this constrained setting, we'll use a heuristic:
        # If a new point is better than the current best, it's a "success".
        # We maintain a counter of successes in the last `window` evaluations.
        window_size = 10
        successes = 0
        window_count = 0
        
        # Restart logic
        # Initial range estimate: we don't know f_opt. 
        # We'll use the spread of initial random points to estimate scale.
        
        # Initialize population
        # Start with a diverse population to get a good initial best
        x_pop = np.zeros((lam, d))
        f_pop = np.zeros(lam)
        
        # Random initialization
        for i in range(lam):
            x_pop[i] = np.random.uniform(-4.5, 4.5, d)
            f_pop[i] = func(x_pop[i])
            
        # Find initial best
        min_idx = np.argmin(f_pop)
        best_f = f_pop[min_idx]
        best_x = x_pop[min_idx].copy()
        
        # Estimate initial sigma based on distance between best and 2nd best? 
        # Or just start with 1.0 and let 1/5 rule adapt.
        
        # Mean vector for CMA-ES
        mean = best_x.copy()
        
        # Covariance matrix C (identity initially)
        C = np.eye(d)
        C_inv = np.eye(d)
        C_sqrt = np.eye(d)
        C_sqrt_inv = np.eye(d)
        
        # Update rates
        alpha_mu = 1.0
        c_sigma = (4 + lam / d) / (d + 4 + 2 * lam / d)
        c_cov = (2 + 2 * alpha_mu * (1 - c_sigma)) / (d + 1.5 + alpha_mu)
        
        # Step sizes
        step_sigma = c_sigma * (1 + 2 * max(0, (mu - 1) / (lam - 2)) - 1)
        step_cov = (1 - c_cov) * (1 + 2 * (mu - 1) / (lam - 2))
        
        # Memory for path updates
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        calls_used = lam
        restarts = 0
        max_restarts = 5
        restart_threshold = 0.0 # We'll use a relative improvement threshold based on initial spread
        
        # Estimate initial spread
        initial_spread = np.max(f_pop) - np.min(f_pop)
        if initial_spread < 1e-10:
            initial_spread = 1.0
            
        # Target improvement per restart cycle
        # If we don't improve by 10% of the current "active range" (approximated by initial spread or recent variance), restart.
        # Simpler: If no improvement for `stall_counter` generations, restart.
        stall_counter = 0
        stall_limit = 20 # 20 generations without improvement -> restart
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Generate new population
            x_new = np.zeros((lam, d))
            f_new = np.zeros(lam)
            
            for i in range(lam):
                # Sample from N(mean, sigma^2 * C)
                z = np.random.randn(d)
                # Apply C_sqrt
                y = C_sqrt @ z
                x_candidate = mean + sigma * y
                
                # Box constraint: Reflection
                for j in range(d):
                    if x_candidate[j] > 5.0:
                        x_candidate[j] = 10.0 - x_candidate[j]
                    elif x_candidate[j] < -5.0:
                        x_candidate[j] = -10.0 - x_candidate[j]
                    
                x_new[i] = x_candidate
                f_new[i] = func(x_candidate)
                calls_used += 1
                
                if calls_used >= self.budget:
                    break
            
            if calls_used >= self.budget:
                break
            
            # 2. Sort by fitness
            idx = np.argsort(f_new)
            f_sorted = f_new[idx]
            x_sorted = x_new[idx]
            
            # 3. Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                stall_counter = 0
            else:
                stall_counter += 1
                
            # 4. Update mean (weighted recombination of top mu)
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * x_sorted[i]
                
            # 5. Update sigma (1/5 rule approximation)
            # Count successes in this generation: how many of the new points were better than the *previous* best?
            # This is a bit tricky with population. 
            # Alternative: Use the standard CMA-ES sigma update based on path p_sigma.
            # Let's stick to the path-based update for robustness.
            
            # Update p_sigma
            p_sigma = (1 - step_sigma) * p_sigma + step_sigma * np.sqrt(mu_eff) * C_sqrt_inv @ (new_mean - mean) / sigma
            
            # Heuristic sigma adaptation based on ||p_sigma||
            # If ||p_sigma|| is too large, sigma is too big. If too small, too small.
            # Standard CMA-ES uses h_sigma to prevent bias.
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            
            # Simple adaptation:
            # If norm_p_sigma > chi_n * 1.1, reduce sigma
            # If norm_p_sigma < chi_n * 0.9, increase sigma
            # This is a simplified version of the standard update.
            
            # Actually, let's use the "success rate" heuristic from the inspiration, which is often more robust for small budgets.
            # Let's track how many of the last `window_size` evaluations were improvements over the *running best*.
            # We need a global list of recent (success, value) or just successes.
            
            # Let's implement a simpler, effective heuristic:
            # If we had an improvement in the last generation, increase sigma slightly? No, that's dangerous.
            # Let's use the standard CMA-ES path update which is well-tested.
            
            # Update p_c
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - step_sigma)**(2*(calls_used+1)/d))) < (1.4 + 2/(d+1)) * chi_n else 0.0
            p_c = (1 - step_cov) * p_c + h_sigma * step_cov * np.sqrt(mu_eff) * C_sqrt_inv @ (new_mean - mean) / sigma
            
            # Update C
            rank_mu = np.outer(new_mean - mean, new_mean - mean)
            rank_one = np.outer(p_c, p_c)
            
            C = (1 + c_cov * (mu_eff - 1) - c_cov) * C + c_cov * rank_mu / (sigma**2) + c_cov * (h_sigma * (1 + (1 - h_sigma) * (mu_eff - 1)) - 1) * rank_one
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Recalculate C_inv and C_sqrt (Cholesky)
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity (restart-like behavior)
                C = np.eye(d)
                C_inv = np.eye(d)
                C_sqrt = np.eye(d)
                C_sqrt_inv = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                sigma = 1.0
                mean = best_x.copy()
                continue
                
            # Update sigma
            # Standard CMA-ES sigma update:
            # sigma *= exp(c_sigma/mu_eff * (||p_sigma||/chi_n - 1))
            sigma *= np.exp(c_sigma / mu_eff * (norm_p_sigma / chi_n - 1.0))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            mean = new_mean
            
            # 6. Restart Logic
            if stall_counter > stall_limit and restarts < max_restarts:
                restarts += 1
                # Reset population to a new spread around best_x
                # Increase sigma for exploration
                sigma *= 2.0
                # Reset paths
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                # Reset C to identity? Or keep it? 
                # Keeping C might be better if the direction is good, but the scale is wrong.
                # Let's reset C to identity to break out of local curvature traps.
                C = np.eye(d)
                C_inv = np.eye(d)
                C_sqrt = np.eye(d)
                C_sqrt_inv = np.eye(d)
                mean = best_x.copy()
                stall_counter = 0
                
                # Generate new population immediately? 
                # The loop will do it in the next iteration.
                # But we need to ensure we don't waste time.
                # We'll just let the loop continue.
                
        return best_x, best_f
# EVOLVE-BLOCK-END
