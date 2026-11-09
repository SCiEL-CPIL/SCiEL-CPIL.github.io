"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        
        # CMA-ES parameters for n=5
        n = self.dim
        lam = 4
        mu = 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        
        mueff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2.0) / (n + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (mueff + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        c_1 = 2.0 / ((n + 1.3)**2 + mueff)
        c_mu = min(1.0 - c_1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0)**2 + mueff))

        def run_cmaes(x0, budget):
            nonlocal best_x, best_f
            if budget <= 0:
                return
            
            x_mean = np.array(x0, dtype=float)
            sigma = 2.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Pre-compute inverse sqrt of C for sampling
            # We will update C and recalculate every step for simplicity (n=5 is small)
            
            for gen in range(budget // lam):
                if time.time() > deadline:
                    return
                
                # Sample
                ys = []
                for k in range(lam):
                    # Sample y ~ N(0, C)
                    # Use Cholesky or eigendecomposition. For n=5, eig is fine and robust.
                    # Actually, we can just use standard normal and multiply by C^0.5
                    # Let's use np.linalg.eigh for stability
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-16, None)
                    C_half = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
                    y = C_half @ np.random.standard_normal(n)
                    y = np.clip(x_mean + sigma * y, -5, 5)
                    ys.append(y)
                
                # Evaluate
                ys_f = np.array([func(y) for y in ys])
                
                # Sort
                idx = np.argsort(ys_f)
                ys = [ys[i] for i in idx]
                ys_f = ys_f[idx]
                
                # Update best
                if ys_f[0] < best_f:
                    best_f = ys_f[0]
                    best_x = ys[0].copy()
                
                # Update CMA-ES state
                # Weighted mean
                x_new = sum(w[i] * ys[i] for i in range(mu))
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (x_new - x_mean))
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Update covariance
                delta_hs = (1 + (1 - c_c) * c_c) * (np.linalg.norm(ps)**2 > (2 + 2.0/n) * (1 - c_c) * n)
                pc = (1 - c_c) * pc + delta_hs * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma
                
                # Rank-one update
                C = (1 - c_1 - c_mu * mueff) * C + c_1 * (pc @ pc.T + (1 - delta_hs) * c_c * (2 - c_c) * C)
                
                # Rank-mu update
                for i in range(mu):
                    zi = (ys[i] - x_mean) / sigma
                    C += c_mu * w[i] * np.outer(zi, zi)
                    
                # Symmetrize
                C = (C + C.T) / 2.0
                
                x_mean = x_new

        # Strategy:
        # 1. Start at 0 (likely local min)
        # 2. Start at random (chance for global)
        # 3. Refine best
        
        # Split budget: 100, 100, 300
        b1 = 100
        b2 = 100
        b3 = self.budget - b1 - b2
        
        run_cmaes(np.zeros(n), b1)
        if time.time() < deadline:
            run_cmaes(np.random.uniform(-4, 4, n), b2)
        if time.time() < deadline:
            # If best_x is None, use random, else use best_x
            x_start = best_x if best_x is not None else np.random.uniform(-4, 4, n)
            run_cmaes(x_start, b3)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
