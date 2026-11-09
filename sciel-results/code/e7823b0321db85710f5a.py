import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        # CMA-ES setup
        lam = max(6, int(4 + 3 * np.log(dim)))
        mu = max(1, lam // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) / (dim + 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        xmean = rng.uniform(-5, 5, dim)
        sigma = 2.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = xmean.copy()
        best_f = func(xmean)
        calls = 1
        
        while calls + lam <= budget and time.time() < deadline - 10:
            # Sample population
            zs = rng.standard_normal((lam, dim))
            X = xmean[None, :] + sigma * (zs @ C.T)
            X = np.clip(X, -5, 5)
            
            # Evaluate
            Fs = np.array([func(xi) for xi in X])
            calls += lam
            
            # Sort
            order = np.argsort(Fs)
            F_sorted = Fs[order]
            X_sorted = X[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
            
            # Update mean
            xmean_old = xmean.copy()
            xmean = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            diff = (xmean - xmean_old) / (sigma + 1e-15)
            pc = (1 - cc) * pc + cc * diff
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma + 1e-15) * diff
            
            # Update covariance
            C = (1 + cc * (2 - cc) - cc * mueff) * C + cc * (pc @ pc.T - (1 - (1 - cc)**2) * C) + \
                cc * cs * damps * (ps @ ps.T - (1 - (1 - cs)**2) * C)
            C = (C + C.T) / 2
            
            # Ensure PSD
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-5 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cc / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Keep xmean in bounds
            xmean = np.clip(xmean, -5, 5)
        
        # Use remaining budget for local refinement if available
        while calls < budget and time.time() < deadline - 2:
            z = rng.standard_normal(dim)
            xi = np.clip(best_x + sigma * 0.5 * z, -5, 5)
            fi = func(xi)
            calls += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        return best_x, best_f
