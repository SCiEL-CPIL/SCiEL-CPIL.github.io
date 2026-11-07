import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        # CMA-ES Parameters
        lam = max(4 + int(3 * np.log(dim)), 4)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Initial state
        x = np.random.uniform(-5, 5, dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        evals = 1
        
        sigma = 2.0
        C = np.eye(dim)
        C_sqr = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff * c_c + c_sigma) / ((dim + 1.3) ** 2 + mueff * c_c)
        c_mu = (2 - c_1) * min(1 - c_sigma, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        while evals + lam <= budget:
            if time.time() > deadline - 5:
                break
                
            # Generate population
            ys = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (C_sqr @ z)
                ys.append(y)
                
            # Evaluate
            fs = []
            for y in ys:
                if evals >= budget:
                    break
                fy = func(y)
                fs.append(fy)
                evals += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            if len(fs) < lam:
                break
            
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            
            # Weighted sum
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]
                
            y_w_norm = y_w / sigma
            
            # Update mean
            x = x + c_sigma * (y_w_norm - p_sigma)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_w_norm
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_w_norm) / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_w_norm
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i_norm = ys_sorted[i] / sigma
                C += c_mu * weights[i] * np.outer(y_i_norm, y_i_norm)
                
            # Update C_sqr
            try:
                C_sqr, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqr = np.eye(dim)
                sigma *= 0.5

        return best_x, best_f
