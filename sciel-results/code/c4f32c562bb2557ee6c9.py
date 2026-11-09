"""Rewritten optimizer for BBOB-style suite using Hybrid CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with adaptive restarts for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # --- Phase 1: Initial Random Sampling ---
        # Sample a few points to get a rough idea of the landscape and initial best
        n_init = min(10, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            if time.time() > deadline:
                return best_x, best_f

        # --- Phase 2: CMA-ES with Restarts ---
        # We will run CMA-ES. If it stagnates, we restart.
        # CMA-ES parameters:
        # lambda: population size. Typically 4 + floor(3 log n)
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # We allocate budget to CMA-ES runs.
        # Each generation costs 'lam' calls.
        # We stop a CMA-ES run if:
        # 1. Budget is exhausted
        # 2. Time limit is reached
        # 3. Stagnation: no improvement for 'stagnation_limit' generations
        # 4. Convergence: step size becomes very small
        
        stagnation_limit = 10 * dim  # Generations without improvement
        max_restarts = 10
        restarts = 0
        
        current_best_f = best_f
        best_x_for_restart = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)

        while calls_made < budget and restarts < max_restarts:
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            # Start center at current best
            m = best_x_for_restart.copy()
            sigma = 1.0  # Initial step size. Could be adaptive based on spread, but 1.0 is safe for [-5,5]
            
            # Initialize covariance matrix to identity
            C = np.eye(dim)
            # Precompute C^{-1/2} and C^{1/2} for sampling
            # Use SVD for stability
            w, V = np.linalg.eigh(C)
            C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
            C_sqrt = V @ np.sqrt(np.maximum(w, 1e-10)) @ V.T
            
            # CMA-ES specific parameters
            # Learning rates
            cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
            cmu = min(1 - cc, 2 * ((dim + 1.3) / dim - 1) / ((dim + 3) * (lam / 2 + 2)))
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            gen = 0
            best_f_in_run = current_best_f
            no_improve_count = 0
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Sample population
                # Standard normal vectors
                z = rng.standard_normal((lam, dim))
                # Transform to search distribution
                # x_i = m + sigma * C^{1/2} * z_i
                # Note: C_sqrt is dim x dim, z is lam x dim
                x_pop = m[None, :] + sigma * (z @ C_sqrt.T)
                
                # Clip to bounds? The problem says func clips, but it's better to keep search within bounds for efficiency
                x_pop = np.clip(x_pop, -5, 5)
                
                # Evaluate population
                f_pop = np.array([func(x) for x in x_pop])
                calls_made += lam
                
                # Update best
                min_idx = np.argmin(f_pop)
                if f_pop[min_idx] < best_f:
                    update_best(x_pop[min_idx], f_pop[min_idx])
                    best_f_in_run = f_pop[min_idx]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Sort population by fitness
                order = np.argsort(f_pop)
                x_sorted = x_pop[order]
                f_sorted = f_pop[order]
                
                # Weights for recombination
                # Default: log-uniform or standard. Let's use standard equal weights for simplicity and robustness
                # mu = lam // 2 (top half)
                mu = lam // 2
                weights = np.zeros(lam)
                weights[:mu] = 1.0 / mu  # Uniform weights for top half
                # Normalize weights
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)
                
                # Update mean
                m_old = m
                m = np.sum(weights[:, None] * x_sorted, axis=0)
                
                # Update path evolution strategies
                # cc and cs are defined above
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) / sigma * C^{-1/2} * (m - m_old)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (m - m_old))
                
                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mueff)
                if hsig:
                    pc = (1 - cs) * pc + cs / sigma * (m - m_old)
                
                # Update covariance matrix
                # C = (1 - mu_sigma) * C + (1 - mu_sigma) * cs * (pc * pc.T) + cmu * sum(weights_i * (y_i * y_i.T))
                # where y_i = (x_i - m_old) / sigma
                y = (x_sorted[:mu] - m_old[None, :]) / sigma
                delta_hsig = 1.0 - hsig
                mu_sigma = 1 - (1 - cs) ** 2 - cmu
                C = (1 - mu_sigma) * C + (1 - mu_sigma) * cs * (pc * pc.T) + cmu * np.sum(weights[:mu, None, None] * (y[:, :, None] * y[:, None, :]), axis=0)
                
                # Re-orthogonalize C to ensure positive semi-definiteness
                # Use SVD again
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-10)
                C = V @ np.diag(w) @ V.T
                C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
                C_sqrt = V @ np.sqrt(w) @ V.T
                
                # Update sigma
                # sigma = sigma * exp(cs / sigma * (||ps|| / E||N(0,I)|| - 1))
                # E||N(0,I)|| approx sqrt(dim) * (1 - 1/(4dim) + ...)
                e_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
                sigma = sigma * np.exp(cs / e_norm * (np.linalg.norm(ps) / e_norm - 1))
                
                # Convergence/Stagnation checks
                gen += 1
                
                # If sigma is too small, we are converged
                if sigma < 1e-8:
                    break
                    
                # If no improvement for too long, restart
                if no_improve_count > stagnation_limit:
                    break
            
            # If we broke out due to stagnation or convergence, prepare for restart
            if calls_made >= budget:
                break
                
            # Restart logic
            restarts += 1
            # New center: slightly perturb the best found so far
            # Or if we are far from best, maybe random?
            # Let's perturb the global best
            if best_x is not None:
                best_x_for_restart = best_x + 0.5 * rng.standard_normal(dim)
                best_x_for_restart = np.clip(best_x_for_restart, -5, 5)
            else:
                best_x_for_restart = rng.uniform(-5, 5, dim)
            
            # Reduce initial sigma for next restart to focus search
            # This is a heuristic. CMA-ES adapts sigma anyway.
            # We can just let CMA-ES re-adapt.
            
        # --- Phase 3: Final Polish ---
        # If we have budget left, do a few local (1+1)-ES steps for fine-tuning
        if calls_made < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
