import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard heuristic for lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        if lam > 2 * dim + 2: lam = 2 * dim + 2
        
        # Initial step size
        sigma = 2.0 # Box is [-5,5], center 0. 2.0 is reasonable.
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        if calls >= budget:
            return best_x, best_f

        while calls < budget:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                # Sample from N(0, C)
                z = np.random.randn(dim)
                # Transform to N(0, C) using Cholesky
                # C is symmetric positive definite
                L = np.linalg.cholesky(C)
                y = L @ z
                x = mean + sigma * y
                # Clip to box
                x_clipped = np.clip(x, -5.0, 5.0)
                
                if calls >= budget:
                    break
                
                f = func(x_clipped)
                calls += 1
                ys.append(y)
                xs.append(x_clipped)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
            
            if len(xs) == 0:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lam))) < (1.4 + 2 / (dim + 1)) * chi_N)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])
            
            C = (1 + c_sigma * c_sigma * mu_eff * (delta_h - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Check convergence
            if np.linalg.norm(mean) > 100: # Divergence check
                break

        return best_x, best_f
