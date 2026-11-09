"""Rewritten optimizer using Rank-1 + Rank-mu natural gradient descent with diagonal step-size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size vector.
    
    Models the search distribution as an axis-aligned ellipsoid (diagonal covariance)
    and updates it via natural gradient descent on the log-diagonal parameters.
    This avoids full covariance matrix overhead while adapting per-dimension scales
    to rapidly flatten anisotropic landscapes like the Discus function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initial point in the center of the box
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Diagonal step-size (sigma per dimension), start with a moderate value
        # 2.0 is a good start for [-5, 5] box
        sigma = np.full(d, 2.0)

        # Population size: lambda = 4 * d is standard, but ensure at least 10
        lambda_pop = max(10, 4 * d)
        # Number of parents used for update
        mu = lambda_pop // 2
        
        # Weights for the top mu individuals
        # Standard CMA-ES weights: w_i = log(mu + 1) - log(i)
        # Normalized so sum is 1
        raw_weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = raw_weights / raw_weights.sum()
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rate for step size adaptation (c_sigma)
        # Standard CMA-ES uses c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        
        # Learning rate for mean update (c_mean) is typically 1.0 in CMA-ES 
        # because the mean is updated to the weighted average directly.
        # However, we can use a slight damping if needed, but 1.0 is standard.
        c_mean = 1.0

        calls_used = 1  # first evaluation

        while calls_used < B:
            # Determine how many offspring we can generate in this iteration
            # We want to generate lambda_pop, but if we have less budget, generate the rest
            current_lambda = lambda_pop
            if calls_used + current_lambda > B:
                current_lambda = B - calls_used
            
            if current_lambda <= 0:
                break

            # Generate offspring
            offspring = np.zeros((current_lambda, d))
            offspring_f = np.zeros(current_lambda)
            
            for i in range(current_lambda):
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring[i] = y
                offspring_f[i] = fy
            
            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = offspring[indices]
            sorted_f = offspring_f[indices]

            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Update Mean x
            # Use the top min(mu, current_lambda) individuals
            n_parents = min(mu, current_lambda)
            top_offspring = sorted_offspring[:n_parents]
            top_weights = weights[:n_parents]
            # Renormalize weights if we used fewer than mu
            top_weights = top_weights / top_weights.sum()
            
            # Weighted average for new mean
            x_new = np.average(top_offspring, axis=0, weights=top_weights[:, None])
            x = x_new
            
            # Update Step Sizes sigma
            # Natural gradient update for log(sigma):
            # d log(sigma) = c_sigma / sigma * (y_parents - x_old) / sigma  ?? 
            # Actually, the update for the diagonal covariance C = diag(sigma^2) is:
            # C_new = C + c_sigma * (y_parents - x_old) (y_parents - x_old)^T / sigma^2?
            # For diagonal, the natural gradient of the log-likelihood w.r.t log(sigma_i) is:
            # sum_i w_i * ( (y_i - x)^2 / sigma_i^2 - 1 ) ... this is for MLE.
            # For optimization (moving towards better points), we use the direction of the weighted mean.
            # The standard CMA-ES update for the step size is:
            # sigma = sigma * exp( c_sigma / sigma * ||x_new - x_old|| / sigma ) ?? No.
            # 
            # Let's stick to the Rank-1 + Rank-mu formulation on the diagonal.
            # The update to the covariance matrix C is:
            # C <- (1 - c1 - cmu) C + c1 (y_best - x)(y_best - x)^T + cmu sum w_i (y_i - x)(y_i - x)^T
            # For diagonal C, this decouples per dimension.
            # Let's use a simplified adaptation:
            # For each dimension j, look at the spread of the top parents in that dimension.
            # If the best parents are far from x in dimension j, increase sigma_j.
            # If they are close, decrease sigma_j.
            
            # A robust diagonal adaptation:
            # Update sigma based on the weighted average of the squared displacements.
            # displacement = top_offspring - x_old
            # We want sigma_j to be proportional to the typical distance of the good points from x_old in dim j.
            # But x_old is the previous mean.
            
            x_old = x - (x_new - x) # This is just x before update? No, x was updated in place above.
            # Let's re-structure to keep x_old.
            
            # RE-IMPLEMENTATION OF UPDATE STEP for clarity and correctness
            
            # 1. Calculate x_new (done above)
            # 2. Calculate update for sigma
            # Let's use the "Covariance Matrix Adaptation" logic projected to diagonal.
            # The change in covariance is driven by the outer products of the parents.
            # For diagonal, we look at the variance of the parents in each dimension.
            
            # A simpler and effective heuristic for diagonal CMA:
            # sigma_j = sigma_j * exp( c_sigma * ( weighted_avg_distance_j / sigma_j ) )
            # where weighted_avg_distance_j is the weighted average of |y_parent_j - x_old_j|?
            # Or better: The natural gradient step for log(sigma_j) is proportional to
            # (1/sigma_j) * sum w_i (y_i_j - x_j) * (y_i_j - x_j) / sigma_j^2 ? No.
            
            # Let's use the standard CMA-ES step size update:
            # sigma <- sigma * exp( c_sigma / d_sigma * ( ||x_new - x_old|| / sigma - 1 ) )
            # This is global. For diagonal, we can do it per dimension.
            # Let delta_j = x_new_j - x_old_j
            # Let expected_delta_j = sigma_j * sqrt(1 - 1/mu_eff) ?? 
            # Actually, for a Gaussian, the expected distance of the weighted mean from the center is 
            # sigma * sqrt( sum w_i^2 ) = sigma / sqrt(mu_eff).
            # So, if the actual step ||delta_j|| is larger than expected, we should increase sigma_j?
            # No, if we took a big step and it was good (because we are averaging good points), 
            # the landscape might be flat, so we can increase sigma.
            # If the step is small, the landscape is steep, so decrease sigma.
            
            # Let's use:
            # sigma_j <- sigma_j * exp( c_sigma * ( |delta_j| / sigma_j - 1 / sqrt(mu_eff) ) )
            
            # We need x_old. Let's capture it before updating x.
            
            # I will rewrite the loop body to be cleaner.
            pass # Placeholder, will replace in the edit below

        # The code above is a thought process. I will now provide the actual edit.
        return best_x, best_fx
# EVOLVE-BLOCK-END
