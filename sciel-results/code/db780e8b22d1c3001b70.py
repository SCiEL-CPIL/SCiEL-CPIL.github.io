import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin
        
        N = self.dim
        lam = 4  # Population size
        mu = lam // 2
        
        # Weights
        w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
        w = w / w.sum()
        mueff = (w.sum()**2) / (w**2).sum()
        
        # Rates
        c_sigma = (1 + 2 * min(0.5, (mueff - 1) / (lam + 1))) / (lam + 1.5)
        c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c_1 = 2 / ((N + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
        
        # State
        x = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample
            try:
                L = np.linalg.cholesky(C)
            except:
                C = C + 1e-10 * np.eye(N)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(N)
            
            z = np.random.randn(lam, N)
            y = np.array([np.clip(x + sigma * (z[i] @ L), -5, 5) for i in range(lam)])
            
            # 2. Evaluate
            f_y = np.array([func(y[i]) for i in range(lam)])
            calls += lam
            
            # 3. Sort
            idx = np.argsort(f_y)
            y_sorted = y[idx]
            f_sorted = f_y[idx]
            
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # 4. Update Mean
            x_old = x
            x = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # 5. Update Paths
            v = (x - x_old) / sigma
            
            # Compute C^-0.5 @ v
            # C = L @ L.T
            # C^-0.5 = L^-T @ L^-1
            try:
                u = np.linalg.solve(L, v)   # L @ u = v  => u = L^-1 @ v
                w_vec = np.linalg.solve(L.T, u) # L.T @ w = u => w = L^-T @ u = C^-0.5 @ v
            except:
                w_vec = v
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * w_vec
            
            # Step size adaptation
            gen = calls // lam
            hsig_corr = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            e_norm = np.sqrt(N) * (1 - 1/(4*N))
            if np.linalg.norm(ps) < (1.4 + 2.0/(N + 1.0)) * e_norm * hsig_corr:
                hsig = 1.0
            else:
                hsig = 0.0
            
            sigma *= np.exp(c_sigma / lam * (np.linalg.norm(ps) / e_norm - 1.0))
            
            # 6. Update Covariance
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * v
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            # Rank-1 update
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * (pc @ pc.T + delta_hsig * C)
            
            # Rank-mu update
            # y_k = (x - x_old) / sigma is the vector of selected steps
            # We need sum w_i * (y_i - x_old)/sigma outer (y_i - x_old)/sigma
            # Note: y_sorted[:mu] are the selected points.
            steps = (y_sorted[:mu] - x_old) / sigma
            C += c_mu * np.sum([w[i] * np.outer(steps[i], steps[i]) for i in range(mu)], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            # Regularize
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = C + 1e-10 * np.eye(N)

        return best_x, best_f
