import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 - mueff + c_sigma) / (n + 4 - mueff)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        rng = np.random.default_rng(42)
        
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        calls = 0
        best_x = m.copy()
        best_f = float('inf')
        
        start_time = time.time()
        deadline = 300.0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            pop = np.zeros((lam, n))
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = m + sigma * (L @ z)
            
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            
            if calls >= budget:
                break
                
            idx = np.argsort(fvals)
            fvals = fvals[idx]
            pop = pop[idx]
            
            m_new = sum(w[i] * pop[i] for i in range(mu))
            
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(n)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv @ (m_new - m)) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            delta_heav = 1 if np.dot(ps, ps) > (1 + 4/(n+1)) * chi_n**2 else 0
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - c_c**2) * delta_heav * np.eye(n))
            
            for i in range(mu):
                y = (pop[i] - m) / sigma
                C += c_mu * w[i] * np.outer(y, y)
            
            C = (C + C.T) / 2
            m = m_new
            m = np.clip(m, -4.9, 4.9)

        return best_x, best_f
