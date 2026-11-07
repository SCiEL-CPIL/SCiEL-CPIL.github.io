import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        d = self.dim
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        # Initial mean
        m = np.zeros(d)
        # Initial step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(d)
        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Adaptation parameters
        cmu = 2 / ((d + 4)**2 + lam)
        cc = (4 - lam / (d + 2)) / ((d + 1) * (lam / 2 + 1))
        cs = (cc + 2) / (d + cc + 5)
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1) - 1)) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        # Covariance matrix adaptation
        # c1 = (2 - mueff/d) / ((d+1.3)^2 + mueff)
        c1 = (2 - mueff/d) / ((d+1.3)**2 + mueff)
        cmu = 2 * (mueff - 2 + 2/mueff) / ((d+2)**2 + mueff)
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1) - 1)) + cs
        
        # Precompute matrix square root of C
        # We need to sample from N(0, C)
        # Use Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
        except:
            C = 0.1 * np.eye(d)
            L = np.linalg.cholesky(C)
            
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Initial evaluation
        # We start with a random point or zero? Zero is often good for BBOB if shifted, but here shift is hidden.
        # Let's start with 0.
        x0 = np.zeros(d)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0
        m = x0.copy()
        
        while calls < self.budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if time.time() > deadline: break
                z = rng.standard_normal(d) # Need rng
                y = L @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
            if len(xs) < lam: break
            
            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            ys_sorted = [ys[i] for i in order]
            
            # Update mean
            m_new = m + sigma * sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update covariance matrix
            # diff = [xs_sorted[i] - m for i in range(mu)]
            diff = np.array([xs_sorted[i] - m for i in range(mu)])
            diff = diff / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - (1 - cs) * cs) * C) + cmu * (diff.T @ (weights[:, None] * diff))
            
            # Update paths
            ps = (1 - cs) * ps + cs / chiN * (np.sqrt(mueff) * L.T @ (m_new - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chiN) < (1.4 + 2 / (d + 1))
            pc = (1 - cc) * pc + float(hsig) * cc / chiN * (np.sqrt(mueff) * (m_new - m) / sigma)
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # Update C and L
            m = m_new
            # Regularize C
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
            except:
                C = 0.1 * np.eye(d)
                L = np.linalg.cholesky(C)
                
            # Stop if sigma is too small
            if sigma < 1e-8:
                break
                
        return best_x, best_f
