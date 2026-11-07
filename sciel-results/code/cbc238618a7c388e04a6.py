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
        # Population size: larger for better exploration of the ridge
        popsize = 10
        
        # Initial point: Origin is a good guess for BBOB
        x = np.zeros(n)
        f_x = func(x)
        
        # Try a few random points to find a better start
        best_x = x.copy()
        best_f = f_x
        for _ in range(5):
            x_try = np.random.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                
        x = best_x
        f_x = best_f
        budget_used = 6
        
        # CMA-ES Parameters
        lambda_ = popsize
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Step size
        sigma = 1.0
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # Momentum vectors
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Chi_n
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma
        
        # Main Loop
        while budget_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            xs = []
            fs = []
            
            # Cholesky decomposition of C
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset
                C = np.eye(n)
                C_sqrt = np.eye(n)
                sigma = 1.0
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                continue
                
            for i in range(lambda_):
                if budget_used >= self.budget:
                    break
                z = np.random.randn(n)
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                budget_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                
            if len(xs) == 0:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Update mean
            x_old = x.copy()
            x = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma
            y_sigma = (x - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for h_sigma
            # Standard: h_sigma = 1 if ||p_sigma/sqrt(1-(1-c_sigma)^2)|| / sqrt(1-(1-c_sigma)^(2*age)) < (1+3/n)*chi_n
            # We simplify by using a fixed threshold or approximating age
            age = budget_used // lambda_
            if age > 0:
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * age))
            else:
                denom = 1.0
            h_sigma = 1 if norm_p_sigma / denom < (1 + 3/n) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            C = (1 + c_1 + c_mu - c_c - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            
            # Rank-mu update
            for i in range(mu):
                C += weights[i] * np.outer(ys[i], ys[i])
                
            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)
            
        return best_x, best_f
