import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        # Use a simple space-filling sequence. Halton is good.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = []
            for i in range(1, count + 1):
                p = []
                # Use first 'dim' primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                for d in range(dim):
                    p.append(halton(i, primes[d]))
                points.append(p)
            return np.array(points)

        n_global = 10
        # Generate Halton points in [0, 1]^n and map to [-5, 5]^n
        h_points = halton_sequence(n_global, n)
        points = h_points * 10.0 - 5.0
        
        values = np.array([func(p) for p in points])
        calls_used = n_global
        
        best_idx = np.argmin(values)
        x_best = points[best_idx].copy()
        f_best = values[best_idx]
        
        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        # Population size lambda
        lam = 10
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_w = np.sum(w) ** 2
        # Step size adaptation
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_w - 1) / n) - 1)) / (n + mu_w + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (n + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_w / n) / (n + 4 + 2 * mu_w / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_w)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((n + 2) ** 2 + mu_w))
        c_cov = c_1 + c_mu
        
        # State variables
        mean = x_best.copy()
        sigma = 1.0  # Initial sigma, might need tuning
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        remaining_budget = budget - calls_used
        generations = remaining_budget // lam
        
        for g in range(generations):
            if time.time() > deadline:
                break
            if calls_used + lam > budget:
                # If we can't fit a full population, evaluate remaining calls individually
                # But CMA-ES needs a population. We can just stop or do a (1+1) step.
                # For simplicity, break if we don't have enough for a full pop.
                # Or, we can evaluate just 1 point to update mean? No, CMA-ES is population based.
                # Let's just break.
                break
                
            # Sample population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails (numerical issues)
                C = np.eye(n)
                L = np.eye(n)
            
            z = np.random.standard_normal((lam, n))
            # y = sigma * L * z
            y = sigma * (z @ L.T)
            x = mean + y
            
            # Clip to bounds
            x = np.clip(x, -5, 5)
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in x])
            calls_used += lam
            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < f_best:
                f_best = f_vals[min_idx]
                x_best = x[min_idx].copy()
                
            # Sort
            order = np.argsort(f_vals)
            x_sorted = x[order]
            f_sorted = f_vals[order]
            
            # Weighted mean
            m_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) / sigma * (L.T @ (m_new - mean))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (g + 1))) - 1))
            
            # Update p_c
            h_sigma = 1 if np.linalg.norm(p_sigma / np.sqrt(1 - (1 - c_sigma) ** (g + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma) ** (g + 1))) else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_w) / sigma * (m_new - mean)
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_cov - c_1 + delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(w[:, None, None] * np.outer(x_sorted[:mu] - mean, x_sorted[:mu] - mean), axis=0) / sigma ** 2
            
            # Update mean
            mean = m_new
            
        return x_best, f_best
