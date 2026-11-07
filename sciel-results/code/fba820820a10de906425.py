# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Hybrid multi-start optimizer for Katsuura function f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling to find a good starting point
        # Katsuura is multimodal, so random sampling helps
        n_init = 50
        for _ in range(n_init):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            evaluate(x)

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: CMA-ES
        # Parameters for CMA-ES
        n = dim
        mu = max(4, int(4 + 3 * math.log(n)))  # Population size for selection
        lam = 2 * mu  # Number of offspring
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = math.log(mu + 0.5) - math.log(i + 1)
        weights /= weights.sum()
        
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_c = (4 + mu_eff) / (n + 4 + mu_eff)
        c_1 = 2 / (n ** 2 + 18)
        c_mu = min(1 - c_1, mu_eff / (n + 2))
        c_sigma = 1 / (n + 3)
        d_sigma = 1 + 2 * max(0, 1 / np.sqrt(n + 1) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # State
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(n)
        p_c = np.zeros(n)
        p_sigma = np.zeros(n)
        gen_count = 0

        # CMA-ES Loop
        while evals_used < budget and time.time() < deadline:
            # Generate population
            z = rng.standard_normal((lam, n))
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                C += 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C_sqrt = np.eye(n)
            
            # Transform
            Y = np.dot(C_sqrt, z.T).T
            X = x_mean + sigma * Y
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                F[i] = evaluate(X[i])
            
            # Check if we ran out of budget mid-generation
            if evals_used >= budget or time.time() > deadline:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Select top mu
            X_sel = X[:mu]
            F_sel = F[:mu]
            
            # Update mean
            x_mean_new = np.dot(weights, X_sel.T).T
            
            # Update covariance matrix
            y_sigma = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            gen_count += 1
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count))) < (1.4 + 2 / (n + 1)) * chi_n
            
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            y_c = (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-mu update
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (X_sel[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            x_mean = x_mean_new
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(x_mean) > 5.0):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
