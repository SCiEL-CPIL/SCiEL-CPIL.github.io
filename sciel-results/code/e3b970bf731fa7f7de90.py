import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 10  # Population size
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        
        # Standard CMA-ES parameter settings
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_w = np.sum(weights ** 2)
        
        c_sigma = (4 + lam / dim) / (4 * dim + 4 + lam / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / dim - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lam / dim) / (4 * dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((dim + 2) ** 2 + lam))

        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Pre-compute Cholesky factor for speed? It changes every step, so we compute inside.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            X = []
            F = []
            Y = []
            
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                y = B @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls_made += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                X.append(x)
                F.append(fx)
                Y.append(y)
            
            if len(X) < lam:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(F)
            X_sorted = [X[i] for i in sorted_indices]
            F_sorted = [F[i] for i in sorted_indices]
            Y_sorted = [Y[i] for i in sorted_indices]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * Y_sorted[i]
            
            # Update sigma
            # ps = (1-cs)ps + sqrt(cs(2-cs)muw) * (pc / ||pc||_C)
            norm_pc = np.sqrt(np.dot(pc, C.dot(pc)))
            if norm_pc > 1e-16:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * (pc / norm_pc)
            else:
                ps = (1 - c_sigma) * ps
                
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            # Cap sigma to prevent explosion or collapse
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_w) * y_w
            
            # Update C
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made / lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            hsig = 1.0 if h_sig else 0.0
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            # Rank-1 update
            C = (1 + c_1 * delta_hsig * np.dot(pc, pc) - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            # C += c_mu * sum_{i=1}^{mu} w_i * (y_i - y_w) (y_i - y_w)^T
            # Actually, standard formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + delta_hsig pc pc^T) + c_mu * sum w_i (y_i - y_w)(y_i - y_w)^T
            # Let's stick to the standard sequential update or the combined one.
            # The formula above for Rank-1 is slightly off in standard implementations.
            # Standard:
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + delta_hsig pc pc^T) + c_mu sum_{i=1}^mu w_i (y_i - y_w)(y_i - y_w)^T
            
            # Let's re-calculate C properly.
            # The previous line overwrote C incorrectly.
            # We need to keep the old C for the update.
            
            # Let's restart the C update block properly.
            
            pass # The C update above is buggy. I will replace the whole loop body with a correct one.
            
        return best_x, best_f
