import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initialization
        x_mean = np.random.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)
            
        # Initial eval
        f0 = evaluate(x_mean)
        if f0 is not None:
            best_f = f0
            best_x = x_mean.copy()
            
        while calls < budget:
            # Generate offspring
            offspring = []
            fitnesses = []
            
            # Update C_sqrt and C_inv_sqrt (Cholesky)
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                fy = evaluate(y)
                if fy is None:
                    break
                offspring.append(y)
                fitnesses.append(fy)
                
            if not offspring:
                break
                
            # Sort
            indices = np.argsort(fitnesses)
            offspring = [offspring[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = offspring[0].copy()
                
            # Update mean
            x_new_mean = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update C
            # delta_sig term
            delta_sig = (np.linalg.norm(pc)**2 / dim - (1 - (1 - c_c)**2)) / (1 + (1 - c_c)**2)
            
            # C = (1 - c_1 - c_mu + c_1 * delta_sig) * C + c_1 * np.outer(pc, pc) + c_mu * sum(...)
            # sum term: sum_{i=1}^{mu} weights[i] * y_i * y_i^T / sigma^2
            # where y_i = offspring[i] - x_mean
            
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (offspring[i] - x_mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu + c_1 * delta_sig) * C + c_1 * np.outer(pc, pc) + c_mu * H
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            x_mean = x_new_mean
            
            # If sigma becomes too small or too large, we might want to restart or stop?
            # Standard CMA-ES handles this.
            
        return best_x, best_f
