import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # State variables
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_x, best_fx
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Search with CMA-ES
        # Use a subset of the budget for global search, say 70%
        global_budget = int(budget * 0.7)
        
        # CMA-ES Implementation
        x = np.zeros(dim)
        fx = safe_func(x)
        m = x.copy()
        sigma = 2.0
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w ** 2)
        
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
        cs = (dim + 1.7) / (dim + 9)
        cmu = min(2 - cc + 2 / (dim - 2), 0.5 / (dim + 1.3))
        damp_sigma = 1 + 2 * max(0, np.sqrt((dim + 1) / (dim + 4) - 1)) + cs
        
        C = np.eye(dim)
        C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21. * dim ** 2))
        
        while calls_made < global_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                y = m + sigma * (C_sqrt_inv.T @ z)
                # Clip before calling to ensure valid input, though func clips too
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = safe_func(y)
                if calls_made >= global_budget:
                    break
            
            if calls_made >= global_budget:
                break
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            m_new = np.sum(w[:, None] * Y[:mu], axis=0)
            
            # Update paths
            C_sqrt = np.linalg.cholesky(C)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv.T @ (m_new - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - lambda_)))) / chi_n < (1 + 2 / (dim + 1)) * 1.4
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cs * damp_sigma + cmu - 1) * C
            C += (1 - cc) * (np.outer(pc, pc) - C)
            rank_mu = np.sum(w[:, None, None] * np.outer(Y[:mu] - m, Y[:mu] - m), axis=0)
            C += cmu * (rank_mu - C)
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / damp_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            m = m_new
            m = np.clip(m, -5, 5)
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                sigma = 1.0

        # Phase 2: Local Refinement
        # Use remaining budget for local search
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None:
            # Use Nelder-Mead or Powell for local refinement
            # Nelder-Mead is good for smooth functions like Rosenbrock
            try:
                res = minimize(
                    safe_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': remaining_budget, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
