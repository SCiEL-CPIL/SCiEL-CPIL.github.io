import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        # Parameters for CMA-ES
        lam = 4 + int(3 * np.log(d))
        if lam < 2: lam = 2
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / d) / ((d + 4) * (2 + mueff / d))
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initial state
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, d)
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        sigma = 2.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Pre-compute eigendecomposition helper
        def update_B_D(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            return B, D
        
        B, D = update_B_D(C)
        
        calls_used = 0
        x_best = None
        f_best = np.inf
        
        # Initial population
        Y = np.empty((lam, d))
        F = np.empty(lam)
        for i in range(lam):
            if time.time() > deadline:
                break
            z = rng.standard_normal(d)
            y = m + sigma * (B @ (D * z))
            y = np.clip(y, -5, 5)
            Y[i] = y
            F[i] = func(y)
            calls_used += 1
            if F[i] < f_best:
                f_best = F[i]
                x_best = y.copy()
                
        if calls_used >= budget or time.time() > deadline:
            return x_best, f_best

        # Main loop
        while calls_used < budget and time.time() < deadline:
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            m_old = m.copy()
            m = np.dot(w, Y[:mu])
            
            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (m - m_old))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_used // lam)))) < (1.4 + 2 / (d + 1)) * np.sqrt(d + 1)
            
            # Update pc
            pc = (1 - cc) * pc + hsig * cc / sigma * (m - m_old)
            
            # Update covariance
            delta = (1 + (2 * (hsig - 0.5)) * cc) - 1
            C = (1 + c1 * delta + cmu) * C
            C -= c1 * np.outer(pc, pc)
            
            # Rank-mu update
            Y_mu = (Y[:mu] - m_old) / sigma
            # Vectorized rank-mu update
            # C -= cmu * sum(wi * outer(Y_mu[i], Y_mu[i]))
            # Using einsum for speed
            C -= cmu * np.einsum('i,ij,ik->jk', w, Y_mu, Y_mu)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update B and D
            B, D = update_B_D(C)
            
            # Update sigma
            # Expected norm of ps if it were standard normal
            sig_norm = np.sqrt(d) * (1 - 1/(4*d))
            sigma *= np.exp(cs * (np.linalg.norm(ps) / sig_norm - 1))
            
            # Generate new population
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(d)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls_used += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = y.copy()
                    
        return x_best, f_best
