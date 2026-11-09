# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard CMA-ES parameters
        mu = max(1, dim // 2)  # Number of parents
        lam = max(2 * mu, dim + 2)  # Population size
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initialize
        # Start near center or use a deterministic start
        x0 = np.zeros(dim)
        # Try a few initial points to find a better starting region?
        # CMA-ES is robust, but for f13 (sharp ridge), starting anywhere is okay.
        # Let's just start at 0.
        
        # Actually, to be safe against "hidden shift", maybe start at 0 is fine as search space is [-5,5].
        # But if the minimum is at 4, starting at 0 with sigma=2 might take time.
        # Let's initialize sigma slightly larger to cover the box quickly.
        sigma = 2.0
        
        x_mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (1 + c_sigma) / (1 - c_sigma)
        cs_1 = (1 - c_sigma) * (1 + c_sigma * (1 - c_c))
        
        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Recalculate c params with mu_eff
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Cholesky of C
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)
            
        best_x = x0.copy()
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)
        
        # Function to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial evaluation
        best_f = evaluate(x0)
        
        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_chol @ z)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in ys:
                fy = evaluate(y)
                fs.append(fy)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Check termination criteria
            # 1. Stagnation
            if len(set(fs_sorted)) == 1:
                break
            # 2. Small sigma
            if sigma < 1e-12:
                break
            # 3. Small spread
            if np.linalg.norm(C_chol) * sigma < 1e-12:
                break
            
            # Update mean
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * ys_sorted[i]
            
            # Update path
            pc = (1 - c_c) * pc + c_c * ((x_mean_new - x_mean) / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_chol @ ((x_mean_new - x_mean) / sigma))
            
            # Update covariance
            delta = 1 if np.sum(ps ** 2) / (1 - (1 - c_sigma) ** (2 * (calls // lam))) < (dim + 1) * 1.4 else 0
            C = (1 - c_1 - c_mu + c_1 * delta) * C
            for i in range(mu):
                v = ys_sorted[i] - x_mean
                C += c_1 * np.outer(v, v)
                C += c_mu * weights[i] * np.outer(v, v)
            
            # Update sigma
            sigma *= np.exp(c_sigma / cs * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            x_mean = x_mean_new
            
            # Update Cholesky
            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, reset C or add identity
                C = C + 1e-6 * np.eye(dim)
                try:
                    C_chol = np.linalg.cholesky(C)
                except:
                    C_chol = np.eye(dim)
                    
            # Safety check for time
            if time.time() > deadline - 5:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
