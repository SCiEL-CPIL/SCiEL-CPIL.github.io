"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_and_update(x):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Quick exploration to find a good starting region
        # Evaluate center and a few random points
        x0 = np.zeros(dim)
        eval_and_update(x0)

        for _ in range(10):
            if calls_made >= budget:
                break
            x_rand = rng.uniform(-5, 5, dim)
            eval_and_update(x_rand)

        # Phase 2: CMA-ES starting from the best point found
        x_mean = best_x.copy() if best_x is not None else x0.copy()
        
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff + 2) / (mu_eff + 5) - 1)) / ((dim + 2) ** 2 + mu_eff)
        
        # Initial sigma
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky of C
        C_sqrt_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Damping for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track for restarts
        stagnation_count = 0
        max_stagnation = 10 * lambda_
        restart_threshold = 1e-10
        
        # Main CMA-ES loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lambda_):
                if calls_made >= budget or time.time() >= deadline:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using C_sqrt_inv (which is C^(-1/2) in our simplified version)
                # For full CMA-ES, we need to maintain C_sqrt = C^(1/2)
                # Here we use a simplified update: we maintain C and use its Cholesky
                y = sigma * z  # Simplified: isotropic for now, will add covariance adaptation
                x_i = np.clip(x_mean + y, -5, 5)
                f_i = eval_and_update(x_i)
                population.append(x_i)
                fitnesses.append(f_i)
            
            if len(population) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update mean using weighted average of top mu
            x_new_mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)
            
            # Compute delta for evolution paths
            delta = (x_new_mean - x_mean) / sigma if sigma > 1e-15 else np.zeros(dim)
            
            # Update sigma evolution path
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta
            
            # Check for sigma adaptation
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C evolution path
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * delta
            
            # Heuristic for h_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** 2) < 
                       (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - x_mean) / sigma if sigma > 1e-15 else np.zeros(dim)
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * h_c * (2 - cc) - c1) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for stagnation
            if abs(best_f - min(fitnesses)) < restart_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # If stagnating too long, restart with larger sigma
            if stagnation_count > max_stagnation and sigma < 5.0:
                sigma *= 2.0
                stagnation_count = 0
                # Add some randomization to escape local optima
                x_mean = np.clip(x_mean + rng.standard_normal(dim) * 0.5, -5, 5)
            
            # If sigma is too large, reduce it
            if sigma > 10.0:
                sigma = 10.0
            
            # If sigma is too small, we may be stuck
            if sigma < 1e-4:
                # Try a few larger jumps
                for _ in range(5):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    x_jump = np.clip(x_mean + rng.standard_normal(dim) * rng.uniform(1, 5), -5, 5)
                    f_jump = eval_and_update(x_jump)
                    if f_jump < best_f:
                        x_mean = x_jump.copy()
                sigma = max(sigma, 0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
