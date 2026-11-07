import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        B = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(N))
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        x = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        cs = (mueff + 2) / (N + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 2) / (N + 2)) - 1) + cs
        cc = (4 - mueff + cs) / (N + 4 + cs)
        c1 = 2 / ((N + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
        
        Cholesky_C = np.linalg.cholesky(C)
        
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        cN = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))
        
        while calls < B:
            # Determine how many points we can evaluate in this generation
            # We need to evaluate lam points. If B - calls < lam, we stop or evaluate partial?
            # Standard CMA-ES requires full population for update.
            # If remaining budget < lam, we break.
            if B - calls < lam:
                break
            
            Y = np.zeros((lam, N))
            for i in range(lam):
                Z = np.random.randn(N)
                Y[i] = x + sigma * Cholesky_C @ Z
            
            Y = np.clip(Y, -5, 5)
            
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                calls += 1
            
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                
            x_new = np.sum(Y[:mu] * w[:, None], axis=0)
            
            y = (x_new - x) / sigma
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * y
            hsig = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2/(N + 1)) * cN else 0
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * y
            
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (pc * pc.T + (1 - hsig) * delta_hsig * C) \
                + cmu * np.sum(w * np.outer(Y[:mu] - x, Y[:mu] - x) / sigma**2, axis=0)
                
            C = (C + C.T) / 2
            try:
                Cholesky_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(N)
                Cholesky_C = np.linalg.cholesky(C)
                
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / cN - 1))
            x = x_new
            x = np.clip(x, -5, 5)
            
        return best_x, best_f
