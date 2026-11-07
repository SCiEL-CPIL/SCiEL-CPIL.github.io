"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 12  # population size
        mu = lam // 2  # number of best solutions for recombination
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial point and sigma
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        def update_covariance():
            nonlocal C, C_inv_sqrt
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - nu) * np.outer(pc, pc) + c_mu * np.sum(weights[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu))
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Compute inverse square root
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T
        
        # First evaluation
        fx_best = func(x_mean)
        x_best = x_mean.copy()
        evals = 1
        
        for gen in range(1, 1000):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                pop[i] = x_mean + sigma * (C_inv_sqrt @ rng.standard_normal(dim))
            
            # Evaluate
            fvals = np.array([func(pop[i]) for i in range(lam)])
            evals += lam
            
            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < fx_best:
                fx_best = fvals[0]
                x_best = pop[0].copy()
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update sigma and ps
            step = x_mean - x_mean_old
            step_norm = np.linalg.norm(step)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * step_norm + 1e-15) * step
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc and C
            if h_sigma:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step / sigma
            else:
                pc = (1 - c_c) * pc
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * ((1 + delta_h) * np.outer(pc, pc) - C) + c_mu * np.sum(weights[i] * np.outer(pop[i] - x_mean_old, pop[i] - x_mean_old) for i in range(mu))
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
