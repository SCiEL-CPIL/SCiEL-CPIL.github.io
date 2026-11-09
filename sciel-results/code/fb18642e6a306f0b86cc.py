"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 10s buffer before 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 9
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Initial state
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        Csqrt = np.eye(d)
        
        # Parameters
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        c1 = (2 + (mueff - 2) / (d + 1)) / ((d + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we can fit a full population
            if calls + lam > budget:
                break

            # 1. Sample
            Y = np.random.randn(lam, d)
            X = m + sigma * (Csqrt @ Y.T).T
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(xi) for xi in X])
            calls += lam
            
            # Update global best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
                
            # 3. Selection
            I = np.argsort(F)
            Ys = Y[I[:mu]]
            
            # 4. Update mean
            m_new = m + sigma * np.sum(w[:, None] * Ys, axis=0)
            
            # 5. Update step size
            y_sigma = (m_new - m) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            
            # 6. Update covariance
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            
            # Delta term
            if np.linalg.norm(pc) / chiN < (1.4 + 2.0 / (d + 1.0)) * (1.0 - cs):
                delta_h = 1.0
            else:
                delta_h = 0.0
                
            # C update
            y_bar = np.sum(w[:, None] * Ys, axis=0)
            Ys_w = Ys * w[:, None]
            rank_mu = Ys_w.T @ Ys - np.outer(y_bar, y_bar)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - delta_h) * cs * (2 - cs)) + cmu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-9 * np.eye(d)
                Csqrt = np.linalg.cholesky(C)
                
            m = m_new

        return best_x, best_f

# EVOLVE-BLOCK-END
