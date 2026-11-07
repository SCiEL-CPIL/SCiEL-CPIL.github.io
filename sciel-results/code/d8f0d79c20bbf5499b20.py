"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for the BBOB Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 295.0  # Safety margin before 330s kill

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def cmaes_iteration(x_mean, sigma, calls_used, max_calls, timeout):
            """Run one CMA-ES iteration. Returns updated x_mean, sigma, calls_used."""
            if calls_used >= max_calls or time.time() - start_time > timeout:
                return x_mean, sigma, calls_used
            
            # Population size for dim=5: lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            
            # Weights for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            # Initial values
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            sigma_0 = sigma
            
            # Max iterations based on remaining budget
            # Each iteration uses lam calls
            max_iters = min(100, (max_calls - calls_used) // lam)
            
            for it in range(max_iters):
                if calls_used >= max_calls or time.time() - start_time > timeout:
                    break
                
                # Sample population
                pop = []
                for i in range(lam):
                    z = np.random.randn(dim)
                    x = x_mean + sigma * (B @ (D * z))
                    pop.append(x)
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if calls_used >= max_calls or time.time() - start_time > timeout:
                        break
                    fval = call_func(x)
                    fvals.append(fval)
                    calls_used += 1
                
                if len(fvals) < lam:
                    break
                
                # Sort by fitness
                idx = np.argsort(fvals)
                pop_sorted = [pop[i] for i in idx]
                fvals_sorted = [fvals[i] for i in idx]
                
                # Update best
                if fvals_sorted[0] < best_fx:
                    best_fx = fvals_sorted[0]
                    best_x = pop_sorted[0].copy()
                
                # Recombination
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * pop_sorted[i]
                
                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (D * (x_new - x_mean)))
                
                # Update evolution path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma
                
                # Update covariance matrix
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1) / dim)) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    delta_corr = (1 - c1) * np.outer(pc, pc)
                    delta_corr += c1 * (1 - cc) * np.outer(ps, ps) / damps
                    delta_corr -= c1 * (1 - cc) * np.outer(ps, ps)
                    C = (1 - c1 - cmu) * C + delta_corr + cmu * np.sum(weights[:mu][:, np.newaxis] * np.outer(pop_sorted[i] - x_mean, pop_sorted[i] - x_mean) for i in range(mu))
                else:
                    C = (1 - c1 - cmu) * C + (1 - c1) * np.outer(pc, pc) + cmu * np.sum(weights[:mu][:, np.newaxis] * np.outer(pop_sorted[i] - x_mean, pop_sorted[i] - x_mean) for i in range(mu))
                
                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs
                
                # Update mean
                x_mean = x_new
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                # Convergence check
                if np.linalg.norm(x_mean - best_x) < 1e-6 * (1 + np.abs(best_x)) and sigma < 1e-6:
                    break
            
            return x_mean, sigma, calls_used

        # Multi-start with progressively smaller step sizes
        # Start from center with large step, then refine
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for sigma_init in step_sizes:
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Start from current best or center
            x_start = best_x.copy() if best_fx < np.inf else np.zeros(dim)
            
            # Run CMA-ES with this initial sigma
            x_mean, sigma, call_count = cmaes_iteration(
                x_start, sigma_init, call_count, budget, timeout
            )
            
            # Update best if CMA-ES found something better
            # (already updated in call_func)
            
            # If we've converged well, stop
            if sigma < 1e-4:
                break
            
            # If we're still making progress, continue with smaller sigma
            # Check if we've used too much budget
            if call_count > budget * 0.8:
                break
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
