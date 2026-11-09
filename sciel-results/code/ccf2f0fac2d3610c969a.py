"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: stratified random search followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Stratified random search to find a good starting point
        n_explore = min(20, max(10, budget // 10))
        
        best_x = None
        best_f = np.inf
        
        # Latin Hypercube Sampling (LHS)
        n = n_explore
        samples = np.zeros((n, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, n + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            perm = rng.permutation(n)
            samples[:, d] = points[perm]
            
        samples = samples * 10 - 5
        
        for x in samples:
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Also try origin
        fx0 = func(np.zeros(dim))
        if fx0 < best_f:
            best_f = fx0
            best_x = np.zeros(dim)
            n_explore += 1

        # Phase 2: CMA-ES
        x = best_x.copy()
        sigma = 1.0
        
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_pop // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((c_sigma + 1) / (c_sigma + 3)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        mean = x.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls_made = n_explore
        
        while calls_made + lambda_pop <= budget:
            # Generate population
            offspring = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            for _ in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                
            # Evaluate
            offspring_f = np.array([func(y) for y in offspring])
            calls_made += lambda_pop
            
            sorted_idx = np.argsort(offspring_f)
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[sorted_idx[i]]
                
            diff = new_mean - mean
            
            # Update ps
            try:
                L = np.linalg.cholesky(C)
                v = np.linalg.solve(L, diff / sigma)
                C_inv_sqrt_diff = np.linalg.solve(L.T, v)
            except np.linalg.LinAlgError:
                C_inv_sqrt_diff = diff / sigma
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_diff
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_pop + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (diff / sigma)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * ps @ ps.T)
            for i in range(mu):
                y_i = (offspring[sorted_idx[i]] - mean) / sigma
                C += c_mu * weights[i] * (y_i @ y_i.T)
                
            C = (C + C.T) / 2
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            best_off_f = offspring_f[sorted_idx[0]]
            if best_off_f < best_f:
                best_f = best_off_f
                best_x = offspring[sorted_idx[0]].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
