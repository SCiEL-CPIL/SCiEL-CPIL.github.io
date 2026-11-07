"""Combined optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Portfolio: coarse random sampling + CMA-ES-like covariance-adapted local search + adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s kill

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_func(x):
            nonlocal best_f, best_x, call_count
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64).clip(-5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return best_f if best_f != float('inf') else 1e10
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Coarse random sampling to find good starting points
        n_random = min(80, budget // 6)
        for i in range(n_random):
            if call_count >= budget or time.time() > deadline - 60:
                break
            x = rng.uniform(-4, 4, dim)
            fx = safe_func(x)

        # Try a few structured points
        special_starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
        ]
        for s in special_starts:
            if call_count >= budget or time.time() > deadline - 60:
                break
            safe_func(s)

        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # Phase 2: CMA-ES-like covariance-adapted local search
        # We'll run a simplified CMA-ES from the best point found so far
        # This adapts the search distribution to the landscape shape
        
        def cma_es_local_search(center, initial_sigma, max_evals):
            nonlocal best_f, best_x, call_count
            
            # CMA-ES parameters
            pop_size = max(4, dim + 2)  # population size
            weights = np.ones(pop_size)
            weights[0] = 2.0  # extra weight for best individual
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_cov = (2 + 2 * mu_eff / dim) / (dim + 3 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Initialize
            mean = center.copy().astype(np.float64)
            sigma = initial_sigma
            C = np.eye(dim)
            C_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(2*dim) - 1/(4*dim**2))
            
            # Precompute Cholesky factor
            C_chol = np.linalg.cholesky(C)
            
            evals_used = 0
            stagnation_count = 0
            best_local_f = best_f
            best_local_x = best_x.copy() if best_x is not None else mean.copy()
            
            for gen in range(max_evals // pop_size + 5):
                if call_count >= budget - 5 or time.time() > deadline - 10:
                    break
                if evals_used >= max_evals:
                    break
                
                # Generate population
                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_new = mean + sigma * C_chol @ z
                    x_new = np.clip(x_new, -5.0, 5.0)
                    offspring.append(x_new)
                
                # Evaluate offspring
                fitness = []
                for x_off in offspring:
                    fx = safe_func(x_off)
                    fitness.append(fx)
                    evals_used += 1
                    if call_count >= budget - 5 or time.time() > deadline - 10:
                        break
                
                if len(fitness) < pop_size:
                    break
                
                # Sort by fitness
                order = np.argsort(fitness)
                sorted_offspring = [offspring[i] for i in order]
                sorted_fitness = [fitness[i] for i in order]
                
                # Update best
                if sorted_fitness[0] < best_local_f:
                    best_local_f = sorted_fitness[0]
                    best_local_x = sorted_offspring[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Update mean
                old_mean = mean.copy()
                for i in range(min(int(mu_eff), len(sorted_offspring))):
                    mean += weights[i] * (sorted_offspring[i] - mean)
                mean = np.clip(mean, -5.0, 5.0)
                
                # Update covariance matrix
                diff = mean - old_mean
                pc = (1 - c_sigma) * pc + c_sigma * diff / (sigma + 1e-30)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_chol.T @ diff / (sigma + 1e-30)
                
                # Rank-one and rank-mu updates
                C = (1 - c_cov) * C + c_cov * (np.outer(ps, ps) + (1 if np.dot(ps, ps) / chi_n**2 < (1 + 2*c_sigma)/(1-c_sigma) else 0) * C)
                
                # Rank-mu update
                for i in range(min(int(mu_eff), len(sorted_offspring))):
                    y_i = (sorted_offspring[i] - old_mean) / (sigma + 1e-30)
                    C += c_cov * weights[i] * np.outer(y_i, y_i)
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except:
                    C = np.eye(dim)
                    C_chol = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # If stagnating too long, reduce sigma
                if stagnation_count > 3:
                    sigma *= 0.5
                    if sigma < 1e-6:
                        break
            
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()
            
            return best_local_x, best_local_f, evals_used

        # Run CMA-ES from the best point
        remaining_budget = budget - call_count
        if remaining_budget > 30 and time.time() < deadline - 30:
            cma_center = best_x.copy()
            cma_sigma = 1.0
            cma_budget = min(remaining_budget - 20, 300)
            cma_es_local_search(cma_center, cma_sigma, cma_budget)
        
        # Phase 3: Additional CMA-ES restarts from diverse points if budget allows
        if call_count < budget - 30 and time.time() < deadline - 20:
            # Try a second CMA-ES from a different region
            alt_center = rng.uniform(-3, 3, dim)
            alt_f = safe_func(alt_center)
            if alt_f < best_f + 50:  # only if reasonably close
                remaining = budget - call_count
                if remaining > 20 and time.time() < deadline - 15:
                    cma_es_local_search(alt_center, 0.8, min(remaining - 10, 150))
        
        # Phase 4: Fine-grained local refinement
        if best_x is not None and call_count < budget - 5 and time.time() < deadline - 5:
            sigma = 0.3
            for step in range(min(100, budget - call_count - 5)):
                if time.time() > deadline - 3:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    sigma *= 1.1
                else:
                    sigma *= 0.7
                if sigma < 0.005:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
