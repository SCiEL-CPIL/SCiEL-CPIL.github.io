"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Number of parents
        mu = lam // 2
        
        # Weights for the parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # c_sigma: learning rate for the step size
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        # c_c: learning rate for the evolution path in the mean space
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        # c_m: learning rate for the covariance matrix
        c_m = (2 + 2 * min(1 - mu_eff / d, 2 / (d + 1))) / (d + 3 + 2 * mu_eff / d)
        
        # Damping for the h_sigma update
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Initial state
        # Start at the center of the box
        x_mean = np.zeros(d)
        # Initial step size. Box is [-5, 5], so range is 10. 
        # A sigma of 1.0 is a good start to explore quickly.
        sigma = 1.0
        # Covariance matrix
        C = np.eye(d)
        # Inverse square root of C (for sampling)
        B = np.eye(d)
        D = np.ones(d)
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        def update_C_B_D(C, B, D):
            # Update B and D from C
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            return B, D

        # Initial update
        B, D = update_C_B_D(C, B, D)
        
        calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Main loop
        while calls < budget:
            # Generate offspring
            offspring = []
            fitnesses = []
            
            # Determine how many offspring we can generate in this generation
            # We can generate up to lam, but we might run out of budget
            n_gen = min(lam, budget - calls)
            
            for i in range(n_gen):
                # Sample from multivariate normal N(x_mean, sigma^2 * C)
                z = np.random.randn(d)
                # Transform z: x = x_mean + sigma * B @ (D * z)
                y = x_mean + sigma * (B @ (D * z))
                
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                offspring.append(y)
                fitnesses.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Sort offspring by fitness
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            idx = np.argsort(fitnesses)
            offspring = offspring[idx]
            fitnesses = fitnesses[idx]
            
            # Use the best mu offspring
            # If n_gen < mu, we use all available
            mu_actual = min(mu, n_gen)
            if mu_actual == 0:
                break
                
            # Recalculate weights if mu changed? 
            # For simplicity and robustness, if we don't have full population, 
            # we just use the available ones with uniform weights or truncated weights.
            # However, standard CMA-ES assumes fixed lambda. 
            # Since budget is 500 and lambda is ~9, we will have many full generations.
            # The last generation might be partial.
            
            if mu_actual < mu:
                # Adjust weights for partial population
                w = np.ones(mu_actual) / mu_actual
            else:
                w = weights[:mu_actual]
                w = w / w.sum() # Renormalize just in case, though they sum to 1
            
            # Calculate weighted mean of the best offspring
            y_w = np.sum(offspring[:mu_actual] * w[:, np.newaxis], axis=0)
            
            # Update mean
            x_mean = x_mean + c_m * (y_w - x_mean)
            
            # Update evolution paths
            # p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B^T @ D^-1 @ (y_w - x_mean) / sigma
            # Note: (y_w - x_mean) is the step taken.
            # We need to transform this step back to the standard normal space.
            # Step in x-space: d_x = y_w - x_mean_old (approx)
            # Actually, the update is:
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B^T @ (D^-1 @ (y_w - x_mean) / sigma)
            
            # Calculate the step in the transformed space
            # y_w - x_mean is the actual step.
            # We need to map it back: z_step = B^T @ (D^-1 @ (y_w - x_mean) / sigma)
            # Note: x_mean was updated, so we should use the difference before update?
            # Standard CMA-ES:
            # x_mean_new = x_mean + c_m * (y_w - x_mean)
            # The step is (y_w - x_mean).
            
            # Let's compute the step vector
            step = y_w - x_mean # This is the step used to update mean
            
            # Transform step to standard normal space
            # z = B^T @ (D^-1 @ step / sigma)
            step_scaled = step / sigma
            step_inv_D = step_scaled / D
            z_step = B.T @ step_inv_D
            
            # Calculate h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(d+1)) * chi_d
            # We need to track the generation or the power term.
            # Let's keep a variable for the power term: h = 1 - (1-c_sigma)^{2*gen}
            # Initially gen=0, h=0.
            # We need to initialize a generation counter or track the decay.
            # Let's add a variable 'gen' or 'power_cc' and 'power_cs'.
            
            # I will refactor the loop to track these powers properly.
            # For now, let's assume standard implementation details.
            
            # To be safe and simple, I will implement the standard update logic carefully.
            
            # Reset variables if needed? No, they persist.
            
            # Calculate h_sigma
            # We need to know the current generation number to compute the denominator.
            # Let's maintain a 'gen' counter.
            
            # I'll rewrite the loop structure to include gen tracking.
            
            break # Break out of this draft loop to rewrite properly

        # RE-IMPLEMENTATION with proper state tracking
        
        # State variables
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        gen = 0
        
        # Pre-compute constants
        # chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        # We need to track the "age" of the paths for the h_sigma check.
        # power_cc = 1.0
        # power_cs = 1.0
        
        power_cc = 1.0
        power_cs = 1.0
        
        while calls < budget:
            n_gen = min(lam, budget - calls)
            if n_gen == 0:
                break
                
            offspring = []
            fitnesses = []
            
            for i in range(n_gen):
                z = np.random.randn(d)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                fitnesses.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            idx = np.argsort(fitnesses)
            offspring = offspring[idx]
            fitnesses = fitnesses[idx]
            
            mu_actual = min(mu, n_gen)
            if mu_actual == 0:
                break
            
            if mu_actual < mu:
                w = np.ones(mu_actual) / mu_actual
            else:
                w = weights[:mu_actual]
            
            y_w = np.sum(offspring[:mu_actual] * w[:, np.newaxis], axis=0)
            
            # Update Mean
            x_mean_old = x_mean.copy()
            x_mean = x_mean + c_m * (y_w - x_mean_old)
            
            # Update Paths
            # p_c
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * B @ (D^-1 @ (y_w - x_mean_old) / sigma)
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^(2*gen+1)) < (1.4 + 2/(d+1)) * chi_d
            
            step = y_w - x_mean_old
            step_scaled = step / sigma
            step_inv_D = step_scaled / D
            z_step = B.T @ step_inv_D # This is the step in standard normal space, used for p_sigma
            # For p_c, we need the step in the "correlation" space, which is B @ (D^-1 @ step / sigma) ?
            # Actually, p_c is updated with the transformed step:
            # p_c = (1 - c_c) p_c + h_c * sqrt(...) * B @ (D^-1 @ (y_w - x_mean) / sigma)
            # Wait, B @ (D^-1 @ z) is the vector in the original space that corresponds to z in the standard normal space?
            # Yes.
            # So z_step is the vector in the "standard normal" coordinates.
            # The vector in the "correlation" coordinates (before B rotation but after D scaling) is D^-1 @ step / sigma.
            # Let's call this v.
            v = step_scaled / D
            # p_c update uses B @ v ? No, p_c is in the same space as x_mean?
            # Standard CMA-ES:
            # p_sigma is in the "standard normal" space (before B and D).
            # p_c is in the "correlation" space? No, p_c is in the space of the mean?
            # Let's check the standard algorithm (Hansen).
            # p_sigma = (1-c_sigma)p_sigma + h_sigma sqrt(c_sigma(2-c_sigma)mu_eff) B^T D^-1 (y_w - x_mean)/sigma
            # p_c = (1-c_c)p_c + h_c sqrt(c_c(2-c_c)mu_eff) B D^-1 (y_w - x_mean)/sigma
            # Wait, B D^-1 ?
            # The sampling is y = x + sigma B D z.
            # So z = B^-1 D^-1 (y-x)/sigma = B^T D^-1 (y-x)/sigma.
            # So z_step is correct for p_sigma.
            # For p_c, the update is usually:
            # p_c = (1-c_c)p_c + h_c sqrt(c_c(2-c_c)mu_eff) B (D^-1 (y_w - x_mean)/sigma)
            # Let's verify.
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + z_step * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            
            # Update p_c
            # Vector for p_c update
            vec_pc = B @ v
            p_c = (1 - c_c) * p_c + vec_pc * np.sqrt(c_c * (2 - c_c) * mu_eff)
            
            # Calculate h_sigma and h_c
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen+1)) < (1.4 + 2/(d+1)) * chi_d
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^(2*gen+1)) < (1.4 + 2/(d+1)) * chi_d
            
            # Update powers
            power_cs *= (1 - c_sigma)**2
            power_cc *= (1 - c_c)**2
            
            # Avoid division by zero
            denom_cs = np.sqrt(max(1 - power_cs, 1e-10))
            denom_cc = np.sqrt(max(1 - power_cc, 1e-10))
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_p_c = np.linalg.norm(p_c)
            
            chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            thresh_sigma = (1.4 + 2/(d+1)) * chi_d
            thresh_c = (1.4 + 2/(d+1)) * chi_d
            
            h_sigma = 1.0 if (norm_p_sigma / denom_cs) < thresh_sigma else 0.0
            h_c = 1.0 if (norm_p_c / denom_cc) < thresh_c else 0.0
            
            # Update Sigma
            # sigma = sigma * exp( c_sigma / chi_d * ( ||p_sigma||/chi_d - 1 ) )
            # Note: This formula is an approximation. The exact one uses the log-normal distribution.
            # Hansen's standard:
            # sigma = sigma * exp( c_sigma / chi_d * ( ||p_sigma||/chi_d - 1 ) )
            # This is the "logarithmic" update.
            
            sigma = sigma * np.exp( (c_sigma / chi_d) * (norm_p_sigma / chi_d - 1) )
            
            # Update Covariance Matrix
            # C = (1 - c_m(2 - c_m + mu_eff * c_sigma / chi_d^2)) C
            #    + c_m ( 1/c_sigma + (1 - 1/c_sigma) h_sigma ) p_sigma p_sigma^T
            #    + c_m (2 - c_m) mu_eff (y_w - x_mean_old)(y_w - x_mean_old)^T / sigma^2 ?
            # No, the second term is:
            # + c_m ( 1/c_sigma + (1 - 1/c_sigma) h_sigma ) (y_w - x_mean_old)(y_w - x_mean_old)^T / sigma^2
            # Wait, the term involving p_sigma is in the standard normal space?
            # No, the covariance matrix C is in the original space.
            # The update is:
            # C = (1 - c_m(2 - c_m + mu_eff * c_sigma / chi_d^2)) C
            #   + c_m ( 1/c_sigma + (1 - 1/c_sigma) h_sigma ) B (p_sigma p_sigma^T) B^T
            #   + c_m (2 - c_m) (y_w - x_mean_old)(y_w - x_mean_old)^T / sigma^2 ?
            
            # Let's use the standard formulation from Hansen's paper:
            # C = (1 - c_m(2 - c_m + mu_eff * c_sigma / chi_d^2)) C
            #   + c_m ( 1/c_sigma + (1 - 1/c_sigma) h_sigma ) B (p_sigma p_sigma^T) B^T
            #   + c_m (2 - c_m) (y_w - x_mean_old)(y_w - x_mean_old)^T / sigma^2
            
            # Note: (y_w - x_mean_old) is the step vector.
            # The term B (p_sigma p_sigma^T) B^T transforms the outer product of p_sigma back to the original space.
            
            # Calculate the coefficients
            coeff_C = 1 - c_m * (2 - c_m + mu_eff * c_sigma / chi_d**2)
            coeff_sigma = c_m * (1 / c_sigma + (1 - 1 / c_sigma) * h_sigma)
            coeff_step = c_m * (2 - c_m)
            
            # Update C
            C = coeff_C * C
            # Add sigma term
            C += coeff_sigma * (B @ (np.outer(p_sigma, p_sigma)) @ B.T)
            # Add step term
            # Note: The step term is (y_w - x_mean_old) (y_w - x_mean_old)^T / sigma^2
            # But wait, the step is in x-space.
            # The covariance update should be consistent with the sampling.
            # Sampling: y ~ N(x, sigma^2 C).
            # So the outer product of the step (in x-space) divided by sigma^2 gives the contribution to C.
            C += coeff_step * (np.outer(step, step) / sigma**2)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update B and D
            B, D = update_C_B_D(C, B, D)
            
            gen += 1

        return best_x, best_f

# EVOLVE-BLOCK-END
