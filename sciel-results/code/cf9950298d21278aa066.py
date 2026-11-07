import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        # Start at center of the box
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute inverse sqrt of C for sampling
        # We will update this every step
        
        # History
        best_x = m.copy()
        best_f = np.inf
        calls = 0
        start_time = time.time()
        deadline = 300.0 # Safety margin for 330s limit
        
        # Function wrapper to count calls and handle time
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial evaluation
        best_f = evaluate(m)
        
        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Calculate remaining budget for this generation
            # We need to ensure we don't exceed budget
            if lam > budget - calls:
                break
            
            # Sample
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            z = np.random.randn(lam, n)
            # Transform to x
            # x = m + sigma * (z @ L.T)
            # Note: z is (lam, n), L is (n, n). z @ L.T is (lam, n).
            # Actually, if C = L L.T, then sqrt(C) = L.
            # x_i = m + sigma * L @ z_i.T
            samples = m + sigma * (z @ L.T)
            
            # Clip samples
            samples = np.clip(samples, -5, 5)
            
            # Evaluate
            fitness = np.array([evaluate(s) for s in samples])
            
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            samples = samples[idx]
            
            # Update mean
            w = weights
            m_new = np.sum(w[:, None] * samples[:mu], axis=0)
            
            # Update covariance matrix
            # Calculate weighted sum of outer products
            ys = samples[:mu] - m
            delta_hs = np.sqrt(mu_eff * (c_1 + c_mu) * (1 - c_1) * np.trace(C))
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * np.sum(w[:, None] * ys, axis=0) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L.T @ (np.sum(w[:, None] * ys, axis=0) / sigma)
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (c_c * (2 - c_c) if np.dot(ps, ps) < (1.4 + c_c) * (n + 1) * c_sigma else 0) * np.eye(n)) + c_mu * np.sum(w[:, None] * (ys * ys.T), axis=0)
            
            # Correction term for ps
            if np.dot(ps, ps) < (1.4 + c_c) * (n + 1) * c_sigma:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (h_sigma * c_c * (2 - c_c)) * np.eye(n))
            
            # Rank-mu update
            C += c_mu * np.sum(w[:, None] * (ys * ys.T), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(2 * (1 - np.exp(-2 * c_sigma))) - 1))
            
            # Update mean
            m = m_new
            
            # Check convergence
            if sigma < 1e-12:
                break
                
        return best_x, best_f
