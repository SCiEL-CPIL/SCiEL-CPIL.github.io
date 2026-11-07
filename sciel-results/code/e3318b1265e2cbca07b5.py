"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 10:
            lam = 10  # Ensure sufficient exploration in 5D

        # Initial mean: Center of the box
        mean = np.zeros(dim)
        
        # Initial sigma: Large enough to explore [-5, 5]
        # The standard deviation of the normal distribution is sigma.
        # To cover [-5, 5] (range 10), 4*sigma ~ 10 => sigma ~ 2.5
        sigma = 2.5

        # Covariance matrix
        C = np.eye(dim)

        # Strategy parameters
        cs = (lam + 2) / (dim + lam / 2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (dim + 1)) - 1)
        cmu = min(1 - cs, 2 / (dim + 1)**2)
        c1 = (2 + 2 * cmu / dim) / ((dim + 1.3)**2 + cs)
        cmu_eff = cmu
        c_sigma = (cs + 2) / (dim + 5)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 0
        best_x = mean.copy()
        best_f = np.inf

        def elapsed_ok():
            return time.time() < deadline

        # Main Loop
        while calls < budget and elapsed_ok():
            # 1. Generate population
            # Sample from N(mean, sigma^2 * C)
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite (should be rare)
                C = 0.9 * C + 0.1 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            samples = []
            for i in range(lam):
                z = np.random.randn(dim)
                x_i = mean + sigma * (L @ z)
                samples.append(x_i)
            
            # 2. Evaluate population
            # Check if we have enough budget for the whole population
            if calls + lam > budget:
                # Evaluate only as many as we can
                remaining = budget - calls
                samples = samples[:remaining]
            
            fvals = np.array([func(s) for s in samples])
            calls += len(samples)
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = samples[idx_best].copy()
                
            # If we ran out of budget, break
            if calls >= budget:
                break

            # 3. Update CMA-ES parameters
            # Sort samples and fitness values
            order = np.argsort(fvals)
            samples = np.array([samples[i] for i in order])
            fvals = fvals[order]
            
            # Update mean
            # Weights for the best mu individuals
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= np.sum(weights)
            
            new_mean = np.sum(weights[:, None] * samples[:mu], axis=0)
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - (1 - cs)**2) * C) + cmu * sum(wi * (yi @ yi.T))
            # where yi = (xi - old_mean) / sigma
            
            # First, update p_c
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * cmu_eff) * (new_mean - mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * cmu_eff) * (new_mean - mean) / sigma
            
            # Update sigma
            sigma *= np.exp((c_sigma / chiN) * (np.linalg.norm(p_sigma) - chiN))
            
            # Update C
            # yi = (xi - mean) / sigma
            Y = (samples[:mu] - mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - (1 - cs)**2) * C) + cmu * np.sum(weights[:, None, None] * (Y[:, :, None] * Y[:, None, :]), axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
