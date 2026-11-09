"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with multi-start restarts for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin from 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        # For dim=5, lambda = 4 + floor(3 * ln(5)) = 4 + 4 = 8
        lambda_ = 8
        # Initial sigma
        # The box is [-5, 5]. A good initial sigma is roughly range/sqrt(dim) or similar.
        # Let's start with a moderate sigma to explore the space.
        sigma = 1.0
        
        # We will run multiple "phases" or restarts
        # Each phase runs CMA-ES until stagnation or budget exhausted
        # We keep the global best
        
        best_f = float('inf')
        best_x = None
        calls_used = 0
        rng = np.random.default_rng(0)
        
        # Number of restarts to attempt
        # We don't want to use too many tiny restarts, but enough to escape local minima
        # Let's say we allow up to 5 restarts if we stagnate early
        max_restarts = 5
        restarts = 0
        
        # Stagnation threshold: if no improvement in best_f for this many generations, restart
        # With lambda=8, 10 generations = 80 calls. 
        # 500 budget / 80 = ~6 restarts possible.
        stagnation_gen = 10
        
        for phase in range(max_restarts + 1):
            if calls_used >= budget:
                break
            if time.time() >= deadline:
                break
            
            # Initialize CMA-ES state
            # Start from a random point, or if we have a best_x, start near it?
            # For global search, starting from a random point is better for the first restart.
            # For subsequent restarts, maybe start from best_x + noise?
            # Let's just start from a random point to ensure diversity.
            if phase == 0:
                x_mean = rng.uniform(-5, 5, dim)
            else:
                # If we have a best, perturb it slightly to explore nearby or jump?
                # Actually, random restart is safer for "global" behavior.
                x_mean = rng.uniform(-5, 5, dim)
                
            sigma = 1.5 # Initial step size
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            # Adaptation parameters
            mu = lambda_ // 2
            weights = np.ones(mu)
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 - cc - 2 * (min(1, (mueff - 2) / (dim + 2)) * (1 - cc - cs))) / \
                 ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            # Initial evaluation of the mean? 
            # CMA-ES usually evaluates the population. 
            # We can evaluate the mean as part of the first population or separately.
            # Let's just run the standard loop.
            
            gen = 0
            local_best_f = best_f # Track best in this phase to detect stagnation relative to global
            # Actually, we want to detect if *this* phase is making progress.
            # So we track the best found *in this phase*.
            phase_best_f = float('inf')
            
            while calls_used < budget:
                if time.time() >= deadline:
                    break
                    
                # Generate population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                # Sample
                z = rng.standard_normal((lambda_, dim))
                pop = x_mean[None, :] + sigma * (z @ L.T)
                
                # Clip to box
                pop = np.clip(pop, -5, 5)
                
                # Evaluate population
                fvals = np.empty(lambda_)
                for i in range(lambda_):
                    if calls_used >= budget:
                        break
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    
                    # Update global best
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                        
                    # Update phase best
                    if fvals[i] < phase_best_f:
                        phase_best_f = fvals[i]
                        
                if calls_used >= budget:
                    break
                    
                # Sort
                idx = np.argsort(fvals)
                pop_sorted = pop[idx]
                fvals_sorted = fvals[idx]
                
                # Update mean
                x_new_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # Weighted sum of outer products
                y = (pop_sorted[:mu] - x_mean[None, :]) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - cc) * C) + \
                    cmu * np.sum(weights[:, None, None] * np.einsum('i,ij,ik->ijk', weights, y, y), axis=0)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update path
                pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (x_new_mean - x_mean) / sigma
                cs = (1 - cs) * cs + cs * np.sqrt(mueff) * (x_new_mean - x_mean) / (sigma * np.linalg.norm(C) ** 0.5)
                # Note: norm(C) is trace(C)/dim? No, usually norm of C is sqrt(trace(C^2)) or similar.
                # Standard CMA-ES uses d_sigma = sqrt((1-(1-cc)^2) * cc / (1-cc)) * (dim + 2 + 2/max(0, (dim-2)-mueff/dim))
                # Let's simplify or use a robust norm.
                # Actually, the standard update for sigma is:
                # sigma = sigma * exp(cs/d_sigma * (||pc||/sqrt(2*cc*(1-cc)) - 1))
                # d_sigma is a constant.
                # Let's use a simplified sigma update to avoid complexity/errors.
                # sigma *= np.exp(cs / 1.0 * (np.linalg.norm(pc) / np.sqrt(2 * cc * (1 - cc)) - 1))
                
                # Simpler sigma update:
                # sigma *= 1.0 + 0.01 * (np.linalg.norm(pc) - np.sqrt(2*cc*(1-cc)))
                # Or just use a fixed rate if we don't want to risk instability.
                # Let's use the standard formula with a fixed d_sigma approximation.
                d_sigma = np.sqrt((1 - (1 - cc)**2) * cc / (1 - cc)) * (dim + 2 + 2 / max(0.1, (dim - 2) - mueff / dim))
                sigma *= np.exp(cs / d_sigma * (np.linalg.norm(pc) / np.sqrt(2 * cc * (1 - cc)) - 1))
                
                # Clamp sigma
                if sigma < 1e-6:
                    sigma = 1e-6
                if sigma > 10.0:
                    sigma = 10.0
                    
                x_mean = x_new_mean
                
                gen += 1
                
                # Check for stagnation
                if gen % stagnation_gen == 0:
                    # If we haven't improved the phase best in the last 'stagnation_gen' generations?
                    # We tracked phase_best_f. We need to track if it improved.
                    # Let's keep a simple counter.
                    pass # We need a counter for "generations since phase improvement"
                
                # To properly implement stagnation, we need a counter.
                # Let's refactor the loop slightly to include a counter.
                # For now, let's just break if sigma is very small or if we've done enough gens.
                if sigma < 1e-5:
                    break
                if gen > 50: # Max generations per phase
                    break

            # End of phase
            # Check if we should restart
            # If we didn't use all budget, and we stagnated (sigma small or gen limit), restart.
            if calls_used < budget and (sigma < 1e-5 or gen >= 50):
                restarts += 1
                # If we found a better solution than the previous best, we might want to keep searching near it?
                # But for global search, random restart is okay.
                continue
            else:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
