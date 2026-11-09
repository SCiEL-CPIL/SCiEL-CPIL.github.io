import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters tuned for d=5
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (d + 4)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initialize
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d) # Not used in simplified version, but keep for structure? No, use ps
        ps = np.zeros(d)
        
        # Precompute constants
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Sample
            z = rng.standard_normal((lam, d))
            sqrt_eigvals = np.sqrt(np.maximum(eigvals, 1e-12))
            x = m + sigma * (z * sqrt_eigvals) @ eigvecs.T
            x = np.clip(x, -5, 5)
            
            # Evaluate
            fvals = np.array([func(xi) for xi in x])
            calls += lam
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = x[min_idx].copy()
                
            # Sort
            order = np.argsort(fvals)
            x_sorted = x[order]
            
            # Update mean
            m_old = m.copy()
            m = m + sigma * (w @ (x_sorted[:mu] - m_old))
            
            # Update covariance adaptation
            y_sigma = (m - m_old) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ y_sigma)
            
            # hsig condition
            # Approximation of E[||ps||]
            # std_ps = sqrt(1 - (1 - c_sigma)^(2 * (calls - lam)))
            # This formula is for the norm.
            # Let's use a simpler heuristic or the standard one.
            # The standard one is:
            # hsig = (||ps|| / sqrt(1 - (1-c_sigma)^(2 * t))) < (1 + 2/(d+1)) * sqrt(1 - (1-c_sigma)^(2 * t))
            # where t is the number of iterations.
            # Here t is roughly (calls - 1) / lam.
            # Let's use a simplified check: if norm(ps) is small, hsig is True.
            # Or just use the standard formula with t = (calls - 1) // lam + 1.
            t_iter = (calls - 1) // lam + 1
            std_ps = np.sqrt(1 - (1 - c_sigma)**(2 * t_iter))
            hsig = (np.linalg.norm(ps) / std_ps) < (1 + 2 / (d + 1)) * std_ps
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(ps, ps)
            y_mu = (x_sorted[:mu] - m_old) / sigma
            C += c_mu * (y_mu.T * w) @ y_mu
            
            # Update step size
            if hsig:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sigma - chi_n))
            
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 100:
                sigma = 100
                
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            
        return best_x, best_f
