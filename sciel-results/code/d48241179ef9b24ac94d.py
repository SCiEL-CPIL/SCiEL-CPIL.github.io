"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        calls_made = 1
        
        # CMA-ES parameters
        max_restarts = 10
        stagnation_threshold = 60
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            x = x_best.copy()
            sigma = 2.0
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Adaptation parameters
            mu = dim // 2 + 1
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(i + 1) - np.log(mu // 2)
            w /= w.sum()
            mueff = 1 / sum(w**2)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            # Population
            pop_size = 2 * dim
            local_best_val = f_best
            local_best_x = x_best.copy()
            stagnation_count = 0
            best_in_restart = f_best
            
            # Generate initial population
            for _ in range(5):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) * sigma
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < local_best_val:
                    local_best_val = fy
                    local_best_x = y
                    x = y
                    if fy < f_best:
                        f_best = fy
                        x_best = y
            
            # Main CMA-ES loop
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(pop_size):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    z = rng.standard_normal(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim) * sigma
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y))
                
                if len(offspring) < mu:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                
                # Compute new mean
                new_x = sum(w[i] * selected[i][1] for i in range(mu))
                
                # Update evolution paths
                diff = new_x - x
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ diff) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chiN) < (1.4 + 2 / (dim + 1))
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * diff / sigma
                
                # Update covariance matrix
                delta_mu = (1 / sigma) * diff if hsig else 0
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc + delta_mu, pc + delta_mu) + (1 - hsig) * (1 - cs) * cs * C)
                for i in range(mu):
                    zi = (selected[i][1] - x) / sigma
                    C += cmu * w[i] * np.outer(zi, zi)
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
                # Update mean
                x = new_x
                
                # Track local best
                if selected[0][0] < local_best_val:
                    local_best_val = selected[0][0]
                    local_best_x = selected[0][1]
                    x = selected[0][1]
                    stagnation_count = 0
                    if local_best_val < f_best:
                        f_best = local_best_val
                        x_best = local_best_x
                else:
                    stagnation_count += 1
                
                # Check for stagnation
                if stagnation_count >= stagnation_threshold:
                    break
                
                # Check if sigma is too small
                if sigma < 1e-8:
                    break
            
            # If we used up budget, break
            if calls_made >= budget:
                break
        
        # Final polish with small steps
        if calls_made < budget and time.time() < deadline:
            polish_sigma = 0.1
            for _ in range(min(30, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(x_best + polish_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < f_best:
                    f_best = fy
                    x_best = y
        
        return x_best, f_best
# EVOLVE-BLOCK-END
