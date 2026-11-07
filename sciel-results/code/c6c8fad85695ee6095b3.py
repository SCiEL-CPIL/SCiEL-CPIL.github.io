import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(d))
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        
        mu_eff = 1.0 / np.sum(w**2)
        
        n = d
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 2) * (2 - mu_eff)) / (n + 2) - 1) + c_sigma
        c_c = (4 - mu_eff + c_sigma) / (n + 4 - mu_eff)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (2 * n) + 1 / (12 * n**2))
        
        m = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        while calls < B:
            if B - calls < lam:
                break
            
            # Compute Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            # Generate population
            z = rng.standard_normal((lam, n))
            x_pop = m + sigma * (z @ L.T)
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.zeros(lam)
            for i in range(lam):
                if calls >= B:
                    break
                calls += 1
                f_pop[i] = func(x_pop[i])
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x_pop[i].copy()
            
            if calls >= B:
                break
            
            # Sort
            idx = np.argsort(f_pop)
            x_pop = x_pop[idx]
            f_pop = f_pop[idx]
            
            # Update mean
            m_old = m
            m = np.sum(w[:, np.newaxis] * x_pop[:mu], axis=0)
            
            # Update path
            v = m - m_old
            try:
                u = np.linalg.solve(L, v)
                y = np.linalg.solve(L.T, u)
            except np.linalg.LinAlgError:
                y = v
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y
            
            g = calls / lam
            threshold = (1.4 + 2 / (n + 1)) * chi_n
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * g)) < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            Y = (x_pop[:mu] - m_old) / sigma
            S = Y.T @ (Y * w)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * C) + c_mu * S
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            C = (C + C.T) / 2
            
            if sigma < 1e-12:
                break

        return best_x, best_f
