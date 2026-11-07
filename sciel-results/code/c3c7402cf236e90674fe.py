"""Optimized program for BBOB f21 (Gallagher 101 peaks) with LHS + CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
    1. Global exploration via Latin Hypercube Sampling (50 points).
    2. Local exploitation via a robust, simplified CMA-ES variant.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return float('inf')
            n_calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Exploration via Latin Hypercube Sampling ---
        # Use 50 samples to cover the space well without wasting too much budget.
        n_lhs = 50
        if n_lhs >= budget:
            n_lhs = budget // 2
        
        # Generate LHS points in [0, 1]^dim
        lhs_indices = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_points = np.array([[i / n_lhs + rng.random() for i in idx] for idx in lhs_indices])
        
        # Map to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate LHS points
        for p in lhs_points:
            if n_calls >= budget or time.time() > deadline - 5:
                break
            x = np.clip(p, -5, 5)
            wrapped_func(x)
        
        # If we haven't made any valid call (should not happen), fallback to center
        if best_x is None:
            x = np.zeros(dim)
            wrapped_func(x)
            if best_x is None:
                return np.zeros(dim), 0.0

        # --- Phase 2: Local Exploitation via Simplified CMA-ES ---
        # Initialize CMA-ES at the best point found in Phase 1
        x0 = best_x.copy()
        
        # CMA-ES Parameters
        pop_size = 30
        # Initial sigma: scale relative to the domain size, but not too large
        # Since we started from a good LHS point, a moderate sigma is appropriate.
        sigma = 1.0 
        
        # CMA-ES State
        mean = x0.copy()
        C = np.eye(dim)
        step_size = 1.0
        # Covariance adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim)
        cs = (mueff + 2.) / (dim + mueff + 5.)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1. / mueff) / ((dim + 2.) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        # Check if we have enough budget for CMA-ES
        if budget - n_calls < pop_size + 5:
            return best_x, best_f

        # Main CMA-ES Loop
        generation = 0
        while n_calls < budget and time.time() < deadline - 5:
            # Generate population
            pop = []
            for i in range(pop_size):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                try:
                    y = mean + sigma * (np.linalg.cholesky(C) @ z)
                except Exception:
                    y = mean + sigma * z
                pop.append(y)
            
            if len(pop) < 2:
                break

            # Evaluate population
            fit = []
            for y in pop:
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                val = wrapped_func(y)
                fit.append((val, y))
            
            if len(fit) < 2:
                break

            # Sort by fitness
            fit.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(fit))):
                new_mean += weights[i] * fit[i][1]
            
            # Update step size (sigma)
            # Check if we should break due to stagnation or small sigma
            if sigma < 1e-10:
                break
                
            # Update covariance matrix and other state
            if len(fit) >= mu:
                # Difference matrix
                D = np.array([fit[i][1] - mean for i in range(mu)])
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chiN * (np.linalg.cholesky(C).T @ (new_mean - mean) / sigma)
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chiN else 0
                pc = (1 - cc) * pc + cc * hsig * (new_mean - mean) / sigma
                
                # Covariance update
                C = (1 + c1 * hsig * (1 - cc) - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / damps * (D.T @ D) / mueff
                
                # Sigma update
                sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                mean = new_mean
            else:
                mean = new_mean

            generation += 1
            
            # Convergence check: if the simplex is too small or no progress
            if generation > 50 and np.linalg.norm(mean - best_x) < 1e-6:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
