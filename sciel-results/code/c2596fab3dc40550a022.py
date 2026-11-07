"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for 5D optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        # Standard CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights for selection
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        cs = (mueff + 1.5) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 - mueff / n) / (n + 4 - mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial values
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        Cinv = np.eye(n)
        Csqrt = np.eye(n)
        Cinv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Cholesky decomposition helper
        def update_chol(C):
            try:
                L = np.linalg.cholesky(C + 1e-16 * np.eye(n))
                return L
            except np.linalg.LinAlgError:
                # Fallback if not positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-16, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        Csqrt = update_chol(C)
        
        # Initial population
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # Pre-generate some random numbers? No, on the fly is fine.
        rng = np.random.default_rng(42)
        
        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (Csqrt @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                calls += 1
                if calls > self.budget:
                    break
            
            if calls > self.budget:
                break
                
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if calls > self.budget:
                    break
                fs[i] = func(ys[i])
                calls += 1
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            xnew = np.sum(w[:, None] * ys[:mu], axis=0)
            
            # Update path
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * (Cinv_sqrt @ (xnew - x) / sigma)
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cs)**((calls // lam + 1) * cs)) <
                    (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * (Cinv_sqrt @ (xnew - x) / sigma) if hsig else (1 - ds) * ps
            
            # Update covariance
            C = (1 + c1 * (2 - c1) - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) - (1 - hsig) * c1 * (2 - c1) * C) \
                + cmu * np.sum(w[i][:, None, None] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky
            Csqrt = update_chol(C)
            Cinv = np.linalg.inv(C)
            Cinv_sqrt = np.linalg.inv(Csqrt)
            
            # Update step size
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            
            # Update mean
            x = xnew
            
        return best_x, best_f
