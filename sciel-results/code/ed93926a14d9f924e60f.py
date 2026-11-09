"""Rewritten optimizer using CMA-ES with adaptive restarts and multiple starts."""
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Clip helper
        def clip(x):
            return np.clip(x, -5.0, 5.0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Initialization
        # Use a small number of random points to find a good starting region.
        init_budget = min(50, self.budget // 5)
        
        if self.budget <= init_budget:
            for i in range(self.budget):
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        for i in range(init_budget):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        n = self.dim
        x_mean = best_x.copy()
        
        # Population size
        lambda_pop = int(4 + 3 * np.log(n))
        lambda_pop = max(lambda_pop, 10)
        
        if remaining_budget < lambda_pop:
            for i in range(remaining_budget):
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # CMA-ES Parameters
        sigma = 0.5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
        
        iteration = 0
        while calls_made < self.budget:
            # Generate population
            pop = np.zeros((lambda_pop, n))
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                pop[i] = x_mean + sigma * (B @ (D * z))
                pop[i] = clip(pop[i])
            
            # Evaluate population
            fvals = np.full(lambda_pop, np.inf)
            for i in range(lambda_pop):
                if calls_made >= self.budget:
                    break
                fvals[i] = func(pop[i])
                calls_made += 1
            
            if calls_made >= self.budget:
                valid_idx = np.where(fvals < np.inf)[0]
                if len(valid_idx) > 0:
                    min_idx = valid_idx[np.argmin(fvals[valid_idx])]
                    if fvals[min_idx] < best_f:
                        best_f = fvals[min_idx]
                        best_x = pop[min_idx].copy()
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            # Update mean
            x_old = x_mean.copy()
            for i in range(mu):
                x_mean += weights[i] * pop[i]
            
            # Update sigma
            x_diff = (x_mean - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * x_diff
            
            # Update sigma based on h_sigma
            h_sigma = 1.0
            # Approximate h_sigma condition
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C-matrix
            p_c_new = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * x_diff
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c_new, p_c_new) + (1 - (1 - c_c)**2) * C)
            
            for i in range(mu):
                y_i = (pop[i] - x_old) / sigma
                C_new += c_mu * weights[i] * np.outer(y_i, y_i)
            
            C = (C_new + C_new.T) / 2
            p_c = p_c_new
            
            if sigma < 1e-10:
                break
            
            iteration += 1
            
        return best_x, best_f
