import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        rng = np.random.default_rng(123)
        calls_used = 0
        
        def check_budget():
            nonlocal calls_used
            return calls_used < self.budget and time.time() < deadline
        
        # CMA-ES implementation
        def cma_es(x0, f0, sigma0, max_calls):
            nonlocal calls_used
            n = self.dim
            lambda_ = 4 + int(3 * np.log(n))  # population size
            mu = lambda_ // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            
            # Adaptation parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            inv_sqr = 1.0
            B = np.eye(n)
            D = np.ones(n)
            
            # Initial evaluation
            best_x = x0.copy()
            best_f = f0
            calls_used += 1
            
            if not check_budget():
                return best_x, best_f
            
            while check_budget():
                # Generate population
                y = np.array([rng.multivariate_normal(np.zeros(n), C) for _ in range(lambda_)])
                z = y / np.linalg.norm(y, axis=1, keepdims=True)
                x = x_mean + sigma * y
                f = np.array([func(np.clip(xi, -5, 5)) for xi in x])
                calls_used += lambda_
                
                if not check_budget():
                    break
                
                # Sort by fitness
                order = np.argsort(f)
                y_sorted = y[order]
                x_sorted = x[order]
                f_sorted = f[order]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                
                # Update mean
                x_new_mean = x_mean + sigma * np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update step-size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * D @ B.T @ (x_new_mean - x_mean) / (sigma * inv_sqr)
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**min(1, calls_used / (n * lambda_)))
                sigma *= np.exp(c_sigma / d_sigma * (hs / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Update covariance matrix
                pc = (1 - c_c) * pc + c_c * (x_new_mean - x_mean) / sigma
                delta_hs = (1 + (1 - c_c) * hs * (2 - c_c)) if hs > (1.4 + 2 / (n + 1)) * (1 - c_c) else 1.0
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(weights[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
                
                # Eigen-decomposition for stability
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.clip(D, 1e-12, None))
                    inv_sqr = 1.0 / D
                except:
                    D = np.ones(n)
                    B = np.eye(n)
                    inv_sqr = 1.0
                
                x_mean = x_new_mean
            
            return best_x, best_f
        
        # Multi-start strategy
        n_starts = 8
        budget_per_start = (self.budget - n_starts) // n_starts
        
        best_x = None
        best_f = np.inf
        
        # Initial diverse starting points
        start_points = []
        # Corner points
        for corner in [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 0, 1, 0, 1)]:
            pt = np.array([c * 4 - 2 for c in corner])
            start_points.append(pt)
        # Random points
        start_points.extend(rng.uniform(-4, 4, (n_starts - len(start_points), self.dim)))
        
        for i, x0 in enumerate(start_points):
            if not check_budget():
                break
            
            f0 = func(np.clip(x0, -5, 5))
            calls_used += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            if not check_budget():
                break
            
            # Run CMA-ES from this start
            x_opt, f_opt = cma_es(x0.copy(), f0, 1.0, budget_per_start)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Final polish with tight CMA-ES
        if check_budget() and best_x is not None:
            x_polish, f_polish = cma_es(best_x.copy(), best_f, 0.1, self.budget - calls_used)
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
