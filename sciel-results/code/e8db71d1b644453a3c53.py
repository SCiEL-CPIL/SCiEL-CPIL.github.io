import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Phase 1: Exploration
        n_explore = 20
        best_x = None
        best_f = np.inf
        
        # Add center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        
        for _ in range(n_explore - 1):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3)**2 + 2)
        cmu = (2 + 2 / (dim + 1) - dim * c1) / ((dim + 1.3)**2 + 2)
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        calls_made = n_explore
        remaining = budget - calls_made
        generation = 0
        
        while remaining >= lambda_pop:
            # Generate offspring
            offspring = []
            for _ in range(lambda_pop):
                z = rng.standard_normal(dim)
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate
            offspring_f = np.array([func(y) for y in offspring])
            calls_made += lambda_pop
            remaining -= lambda_pop
            generation += 1
            
            sorted_idx = np.argsort(offspring_f)
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[sorted_idx[i]]
            
            # Update ps
            diff = new_mean - mean
            try:
                C_inv = np.linalg.inv(C)
                d_sq = diff @ C_inv @ diff
                d = np.sqrt(max(0, d_sq))
            except np.linalg.LinAlgError:
                d = np.linalg.norm(diff) / sigma
                
            if d > 1e-10:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff / (sigma * chi_n)
            else:
                ps = (1 - cs) * ps
                
            # h_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * generation))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            # Update C
            y_c = diff / sigma
            y_c_outer = np.outer(y_c, y_c)
            ps_outer = np.outer(ps, ps)
            
            C_update = (1 - c1 - cmu) * C
            C_update += c1 * (y_c_outer + (1 - hsig) * cs * (2 - cs) * ps_outer)
            
            for i in range(mu):
                y_i = (offspring[sorted_idx[i]] - mean) / sigma
                C_update += cmu * weights[i] * np.outer(y_i, y_i)
                
            C = C_update
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Update best
            if offspring_f[sorted_idx[0]] < best_f:
                best_f = offspring_f[sorted_idx[0]]
                best_x = offspring[sorted_idx[0]].copy()
                
        # If there are remaining calls, use them for local search or just more CMA-ES steps?
        # If remaining < lambda_pop, we can't do a full CMA-ES step.
        # We can do a few random steps around the best point.
        if remaining > 0:
            for _ in range(remaining):
                # Local random step
                x_new = best_x + rng.standard_normal(dim) * sigma * 0.1
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()

        return best_x, best_f
