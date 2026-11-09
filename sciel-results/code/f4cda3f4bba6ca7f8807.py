import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Global Search
        # Use CMA-ES with large sigma
        x_best, f_best = self._cma_es(func, np.zeros(self.dim), 5.0, 400, seed=42)
        
        # Phase 2: Local Refinement
        # Use CMA-ES with smaller sigma
        x_best, f_best = self._cma_es(func, x_best, 1.0, 100, seed=42)
        
        return x_best, f_best

    def _cma_es(self, func, x0, sigma_init, budget, seed=0):
        n = self.dim
        rng = np.random.default_rng(seed)
        
        # Hyperparameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        cc = (4 + lam/n) / (n + 4 + 2*lam/n)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(c1, 2 * (mueff - 2 + 2/mueff) / ((n + 2)**2 + mueff))
        
        # State
        m = np.array(x0, dtype=float)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Cholesky factor
        L = np.eye(n)
        
        evals = 0
        best_x = m
        best_f = func(m)
        evals += 1
        
        # Pre-compute expected norm of N(0, I)
        # E||N(0,I)|| = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
        # For n=5, it's approx 2.5
        # We can just use np.sqrt(n) as a rough estimate or calculate it.
        # Let's calculate it properly.
        from math import gamma, sqrt
        enorm = sqrt(2) * gamma((n + 1) / 2) / gamma(n / 2)
        
        while evals < budget:
            # Generate samples
            ys = []
            for _ in range(lam):
                z = L @ rng.standard_normal(n)
                y = m + sigma_init * z
                ys.append(y)
            
            # Evaluate
            fs = [func(y) for y in ys]
            evals += lam
            
            # Sort
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0]
                
            # Update Mean
            m_old = m.copy()
            m = np.zeros(n)
            for i in range(mu):
                m += w[i] * ys_sorted[i]
            
            # Update PC
            inv_sigma = 1.0 / sigma_init
            pc = (1 - cc) * pc + cc * inv_sigma * (m - m_old)
            
            # Update PS
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sigma * (L.T @ (m - m_old))
            
            # Update Sigma
            sigma_init *= np.exp(cs / ds * (np.linalg.norm(ps) / enom - 1))
            
            # Update C
            # Rank 1 update
            rank1 = np.outer(pc, pc)
            # Rank mu update
            v = np.array([ys_sorted[i] - m_old for i in range(mu)])
            rankmu = np.zeros((n, n))
            for i in range(mu):
                rankmu += w[i] * np.outer(v[i], v[i])
            
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Symmetrize and check PD
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(n)
                L = np.linalg.cholesky(C)
                
        return best_x, best_f
