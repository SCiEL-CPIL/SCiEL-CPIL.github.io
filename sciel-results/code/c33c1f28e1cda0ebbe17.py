import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        x_best = None
        f_best = np.inf
        
        # CMA-ES Parameters
        N = 10
        mu = N // 2
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        restart_count = 0
        
        # Phase 1: Rapid exploration with diverse starting points
        # Evaluate a diverse set of points to quickly find the basin of attraction
        exploration_budget = min(50, total_budget // 10)
        
        # Generate diverse starting points using Latin Hypercube Sampling
        def latin_hypercube_sample(n, dim, rng):
            """Generate Latin Hypercube samples in [-5, 5]^dim"""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = -5 + 10 * (perm + rng.random(n)) / n
            return points
        
        # First, evaluate a diverse set of points
        n_explore = exploration_budget
        explore_points = latin_hypercube_sample(n_explore, dim, rng)
        
        for i in range(n_explore):
            if used_calls >= total_budget or time.time() > deadline:
                break
            x = explore_points[i]
            f = func(x)
            used_calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
        
        # Phase 2: CMA-ES exploitation from the best found point
        # Use the best point from exploration as the starting point
        if x_best is None:
            x_best = rng.uniform(-5, 5, dim)
            f_best = func(x_best)
            used_calls += 1
        
        # Run multiple CMA-ES restarts with decreasing sigma
        # Start with moderate sigma to explore around the best point
        init_sigmas = [1.5, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02]
        
        # Track the best point across all runs
        global_best_x = x_best.copy()
        global_best_f = f_best
        
        while used_calls < total_budget and time.time() < deadline:
            # Choose starting point and sigma
            # For early restarts, perturb the global best
            # For later restarts, use random points
            if restart_count < 3:
                # Perturb the global best with increasing noise
                noise_scale = 0.5 * (restart_count + 1)
                x_start = global_best_x + rng.standard_normal(dim) * noise_scale
                x_start = np.clip(x_start, -5, 5)
            else:
                # Random restarts for diversity
                x_start = rng.uniform(-5, 5, dim)
            
            # Select sigma based on restart count
            if restart_count < len(init_sigmas):
                sigma_init = init_sigmas[restart_count]
            else:
                sigma_init = rng.uniform(0.01, 0.1)
            
            # Run CMA-ES from x_start
            mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            min_sigma = 1e-14
            # Shorter stagnation limit to allow more restarts
            stagnation_limit = 15
            
            run_calls = 0
            best_run_x = x_start.copy()
            best_run_f = np.inf
            
            # Evaluate initial point
            if used_calls + 1 > total_budget:
                break
                
            f_mean = func(x_start)
            used_calls += 1
            run_calls += 1
            
            if f_mean < f_best:
                f_best = f_mean
                x_best = x_start.copy()
                global_best_x = x_start.copy()
                global_best_f = f_best
            best_run_f = f_mean
            best_run_x = x_start.copy()
            
            no_improve_count = 0
            gen_count = 0
            
            # Main CMA-ES Loop
            while used_calls + N <= total_budget and time.time() < deadline:
                if sigma < min_sigma:
                    break
                    
                gen_count += 1
                
                # Sample N offspring
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    
                B = rng.standard_normal((N, dim)) @ L.T
                Y = mean + sigma * B
                Y = np.clip(Y, -5, 5)
                
                # Evaluate
                F = np.empty(N)
                for i in range(N):
                    if used_calls >= total_budget or time.time() > deadline:
                        break
                    F[i] = func(Y[i])
                    used_calls += 1
                    run_calls += 1
                    
                    if F[i] < best_run_f:
                        best_run_f = F[i]
                        best_run_x = Y[i].copy()
                        if F[i] < f_best:
                            f_best = F[i]
                            x_best = Y[i].copy()
                            global_best_x = Y[i].copy()
                            global_best_f = F[i]
                
                # If we broke out of evaluation loop early, break outer loop
                if used_calls + N > total_budget or time.time() > deadline:
                    break
                    
                # Sort and select
                I = np.argsort(F)
                Y_sorted = Y[I]
                F_sorted = F[I]
                
                # Stagnation check
                if F_sorted[0] < best_run_f - 1e-12:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > stagnation_limit:
                    break
                
                # Update mean
                mean_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update sigma and p_sigma
                y_sigma = (mean_new - mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (run_calls + 1))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (run_calls + 1))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = (Y_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                    
                C = (1 + c_1 * h_c - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                mean = mean_new
                
                # Convergence check
                if np.all(np.abs(mean - x_start) < 1e-6):
                    break
                    
            restart_count += 1

        return x_best, f_best
