import numpy as np

class Optimizer:
    """CMA-ES optimizer for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        # CMA-ES parameters
        lam = 10  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        x0 = np.zeros(n)
        sigma0 = 1.0
        
        m = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        cs = (1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)) * cc
        ds = 1 + cs + 2 * max(0, (mueff - 1) / (n + 1) - 1)
        
        calls = 0
        best_x = x0.copy()
        best_f = func(best_x)
        calls += 1
        
        while calls < self.budget:
            # Generate population
            ys = []
            fs = []
            
            # Calculate Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
                
            for i in range(lam):
                z = np.random.randn(n)
                y = m + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
                
                if calls >= self.budget:
                    break
            
            if len(ys) == 0:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            w = weights[:len(ys)]
            m_new = np.sum(w[:, None] * ys, axis=0)
            
            # Update step size control
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * (m_new - m) / sigma
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * cc) * C @ (m_new - m) / sigma
            
            # Check if ps is too large
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(np.trace(C))
            hsig = 1 if norm_ps / expected_norm < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - ds)**2) else 0
            
            # Update sigma
            sigma *= np.exp(1 / ds * (norm_ps / expected_norm - 1))
            sigma = np.clip(sigma, 1e-8, 1e8)
            
            # Update covariance matrix
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (1 - cc) * C) + cmu * np.sum(w[:, None] * np.outer(ys - m, ys - m), axis=0)
            
            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            m = m_new
            
        return best_x, best_f
