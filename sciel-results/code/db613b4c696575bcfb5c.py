import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls
            if calls >= B:
                return np.inf
            calls += 1
            val = func(x)
            return val

        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(d))
        sigma = 2.0
        m = np.zeros(d)
        C = np.eye(d)
        
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1) - 1)) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        ps = np.zeros(d)
        pc = np.zeros((d, d))
        
        # Main Loop
        while calls < B:
            # Generate Population
            L = np.linalg.cholesky(C)
            X = np.zeros((lam, d))
            for i in range(lam):
                z = rng.standard_normal(d)
                X[i] = m + sigma * (L @ z)
                X[i] = np.clip(X[i], -5, 5)
                
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= B:
                    break
                F[i] = eval_func(X[i])
            
            if calls >= B:
                break
                
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update Best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Select
            x_sel = X[:mu]
            
            # Update Mean
            m_old = m.copy()
            m = np.sum(w[:, np.newaxis] * x_sel, axis=0)
            
            # Update Path ps
            v = (m_old - m) / sigma
            # Compute C^{-1/2} v
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            step = C_inv_sqrt @ v
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * step
            
            # Update Sigma
            if np.linalg.norm(ps) / chiN > 1.4 + 2 / (d + 1):
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
            # Update Path pc
            u = step
            pc = (1 - cs) * pc + cs * (1 + c1 - cmu) * np.outer(u, u)
            
            # Update Covariance
            rank1 = pc * (1 + (1 - cs) * cs) + (pc - np.eye(d))
            
            rankmu = np.zeros((d, d))
            for i in range(mu):
                diff = x_sel[i] - m
                rankmu += w[i] * np.outer(diff, diff)
                
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            C = (C + C.T) / 2
            
            # Check for PD
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(d)
                
            # Stopping Criteria
            if sigma < 1e-12:
                break
            if F[-1] - F[0] < 1e-8:
                break

        return best_x, best_f
