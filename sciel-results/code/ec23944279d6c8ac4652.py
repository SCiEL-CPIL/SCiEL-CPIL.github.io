import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        if n <= 5:
            lam = 8
        else:
            lam = 4 + int(3 * np.log(n))
            
        mu = lam // 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= np.sum(w)
        
        mueff = 1.0 / np.sum(w**2)
        
        # Control parameters
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(0.5, (2 - c1 + mueff / n) / ((n + 2) ** 2 + mueff))
        
        # Initial state
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        
        best_x = m
        best_f = np.inf
        calls = 0
        
        rng = np.random.default_rng(42)
        
        while calls < budget:
            # If not enough budget for a full generation, do single steps
            if budget - calls < lam:
                for _ in range(budget - calls):
                    z = rng.standard_normal(n)
                    y = m + sigma * (B @ z)
                    x = np.clip(y, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                break

            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * (B @ z)
                x = np.clip(y, -5, 5)
                X[i] = x
                F[i] = func(x)
                calls += 1
                
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Update Mean
            m_new = np.dot(w, X_sorted[:mu])
            
            # Update Sigma
            diff = (m_new - m) / sigma
            v = np.linalg.solve(B, diff)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            
            sigma *= np.exp((cs / chi_n) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc and C
            diff_m = m_new - m
            v_pc = np.linalg.solve(B, diff_m)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * v_pc
            
            hsig = 1.0 if (np.linalg.norm(ps) / chi_n) < (1.4 + 2 / (n + 1)) * (1 + cs) else 0.0
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + hsig * (1 - (1 - cc) ** 2) * np.eye(n))
            
            for i in range(mu):
                y_i = X_sorted[i] - m
                C += cmu * w[i] * np.outer(y_i, y_i)
                
            m = m_new
            
            # Update B (Cholesky)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.linalg.cholesky(C + 1e-9 * np.eye(n))
                
            if sigma < 1e-10:
                break

        return best_x, best_f
