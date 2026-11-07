import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, sigma0, budget_alloc):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return
            
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2
            # Log weights
            w = np.log(np.arange(mu, 0, -1) + 1) - np.log(mu)
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_m = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_c = (4 + mu_eff / dim) / (dim + 12)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            hs = np.sqrt((1 + 2/(dim+1)) * (1 - c_c) + c_c)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            gen = 0
            while calls < budget - budget_alloc + 10: # Reserve some calls
                # Sample
                X = []
                for i in range(lambda_):
                    z = np.random.multivariate_normal(np.zeros(dim), C)
                    x = m + sigma * z
                    X.append(x)
                
                # Evaluate
                F = []
                for x in X:
                    if calls >= budget:
                        break
                    F.append(safe_func(x))
                
                if len(F) < lambda_:
                    break
                    
                # Sort
                idx = np.argsort(F)
                X = [X[i] for i in idx]
                F = [F[i] for i in idx]
                
                # Update mean
                m_new = m.copy()
                for i in range(mu):
                    m_new += w[i] * X[i]
                
                # Update step size
                sigma_old = sigma
                # Normal distribution check (simplified)
                # Calculate ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.sqrt(np.trace(C)) * (m_new - m) / sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                pc = (1 - c_c) * pc + c_c * (m_new - m) / sigma
                C = (1 + c_c * c_m * (2 - c_c) - c_m) * C
                for i in range(mu):
                    C += c_m * w[i] * np.outer(X[i] - m, X[i] - m)
                C += c_m * c_c * (2 - c_c) * np.outer(pc, pc)
                
                m = m_new
                gen += 1
                
                # Check convergence
                if np.linalg.norm(m_new - m) < 1e-10 or sigma < 1e-10:
                    break

        # Multi-start
        n_starts = 3
        budget_per_start = budget // n_starts
        starts = [np.zeros(dim), np.ones(dim)*2, -np.ones(dim)*2]
        # Add random starts
        rng = np.random.default_rng(42)
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4, 4, dim))
            
        for i in range(n_starts):
            if calls >= budget:
                break
            x0 = starts[i]
            sigma0 = 1.0
            cma_es(x0, sigma0, budget_per_start)
            
        return best_x, best_f
