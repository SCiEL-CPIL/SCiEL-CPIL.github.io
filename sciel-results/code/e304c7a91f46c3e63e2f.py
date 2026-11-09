import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0

        n = self.dim
        lam = 4 + int(3 * np.log(n))
        sigma = 2.0
        m = np.zeros(n)
        C = np.eye(n)
        
        cm = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 2)) - 1) / (n + 1)
        cs = (n + 1.3) / (n + lam / 2 + 3)
        ds = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 2)) - 1) + cs
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c1 = (2 + 2.0 / (n + 1.3)) / ((n + 2)**2 + 2 + 2.0 / (n + 1.3))
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1. / mu_eff)) / ((n + 2)**2 + 2))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = m.copy()
        best_f = np.inf
        calls = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            L = np.linalg.cholesky(C)
            
            fvals = np.zeros(lam)
            y = np.zeros((lam, n))
            
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = np.random.randn(n)
                y[i] = m + sigma * (L @ z)
                # Clip to box
                y[i] = np.clip(y[i], -5, 5)
                fvals[i] = func(y[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = y[i].copy()
            
            if calls >= self.budget:
                break
            
            # Sort
            order = np.argsort(fvals)
            y_sorted = y[order]
            
            # Update mean
            m_new = m + cm * np.sum(w[:mu] * (y_sorted[:mu] - m), axis=0)
            
            # Update paths
            norm_ps = np.linalg.norm(ps)
            c_sigma = np.sqrt(cs * (2 - cs) * mu_eff)
            hsig = norm_ps / np.sqrt(1 - (1 - cs)**(-2 * (calls + 1) / lam)) < (1 + 2 / (n + 1)) * np.sqrt((mu_eff - 1) / (mu_eff - 1) + 1)
            
            pc = (1 - cs) * pc + hsig * cs * c_sigma * (m_new - m) / sigma
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - m) / sigma
            
            delta_hsig = (1 - hsig) * cs * (1 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + delta_hsig * C) \
                + cmu * np.sum(w[:mu] * np.outer(y_sorted[:mu] - m, y_sorted[:mu] - m), axis=0)
            
            m = m_new
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Regularize C
            C = (C + C.T) / 2.0
            # Add small diagonal if not PD
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(n)

        # Polish
        remaining = self.budget - calls
        if remaining > 0:
            x = best_x
            fx = best_f
            sigma_local = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * np.random.randn(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
