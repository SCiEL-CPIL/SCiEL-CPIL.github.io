import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES Parameters
        pop_size = 4 + int(3 * np.log(dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 2)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs

        # State
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        
        for iteration in range(2000):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Generate
            try:
                chol_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                chol_C = np.eye(dim)
                sigma = 2.0
                ps = np.zeros(dim)
                pc = np.zeros((dim, dim))
                
            Y = np.random.randn(pop_size, dim)
            X = np.array([mean + sigma * (chol_C @ Y[i].T).T for i in range(pop_size)])
            
            # Evaluate
            F = np.array([evaluate(X[i]) for i in range(pop_size)])
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update Mean
            mean_old = mean
            mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update Ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (chol_C @ (mean - mean_old).T).T
            
            # Update Pc
            norm_ps = np.linalg.norm(ps)
            # hsig condition
            hs = (norm_ps / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            pc = (1 - cc) * pc + hs * cc * (mean - mean_old)[:, np.newaxis] * (mean - mean_old)[np.newaxis, :]
            
            # Update C
            delta_hs = (1 - hs) * cc * (2 - cc)
            C = (1 + c1 + cmu - c1 - delta_hs) * C \
                + c1 * (pc @ pc.T - C) \
                + cmu / sigma**2 * np.sum(weights[:, np.newaxis, np.newaxis] * (X[:mu] - mean_old)[:, np.newaxis, :] * (X[:mu] - mean_old)[:, :, np.newaxis], axis=0)
                
            # Update Sigma
            sigma = sigma * np.exp(cs / ds * (norm_ps / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) - np.sqrt(dim * (1 - 1 / (dim + 1)))))
            
            if sigma < 1e-12:
                break
            if sigma > 100:
                sigma = 100

        return best_x, best_f
