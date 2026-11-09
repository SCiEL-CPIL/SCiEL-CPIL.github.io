"""CMA-ES with adaptive population size for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizes Katsuura f23 using CMA-ES with adaptive population size.
    1. Initial random start point.
    2. Run CMA-ES with population size starting at 4 and doubling on stagnation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Phase 1: Initial Point ---
        x0 = rng.uniform(lower, upper, size=dim)
        x0, f0 = evaluate(x0)
        
        if calls_made >= budget or check_deadline():
            return x_global_best, f_global_best

        # --- Phase 2: CMA-ES with Adaptive Population Size ---
        # CMA-ES state
        x_mean = x0.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        p_sigma = np.zeros(dim)  # Cumulation for sigma control
        p_c = np.zeros(dim)  # Cumulation for covariance control
        
        # Control parameters
        mu = 2  # Number of recombination weights (top mu)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Population size schedule
        lam = 4  # Initial population size
        lam_max = 16  # Maximum population size
        lam_min = 4  # Minimum population size
        
        best_f = f0
        no_improve = 0
        max_no_improve = 10  # Generations without improvement before adapting lam
        
        while calls_made < budget and not check_deadline():
            # Update mu to be half of lam, at least 2
            mu = max(2, lam // 2)
            # Recalculate weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                if calls_made >= budget or check_deadline():
                    break
                    
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y, fy = evaluate(y)
                ys.append(y)
                fs.append(fy)
            
            if len(ys) == 0:
                break
                
            # Sort by function value
            indices = np.argsort(fs)
            ys_sorted = [ys[i] for i in indices]
            fs_sorted = [fs[i] for i in indices]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                no_improve = 0
            else:
                no_improve += 1
                
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * ys_sorted[i]
            x_mean = y_w
            
            # Update sigma control
            y_sigma = (L.T @ (y_w / sigma))
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian potential
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1)) else 0.0
            
            sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
            
            # Update covariance control
            y_c = (L.T @ (y_w / sigma))
            p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
            for i in range(mu):
                y_i = (L.T @ (ys_sorted[i] / sigma))
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
            
            # Adapt population size
            if no_improve >= max_no_improve:
                if lam < lam_max:
                    lam = min(lam * 2, lam_max)
                elif lam > lam_min:
                    lam = max(lam // 2, lam_min)
                no_improve = 0
                max_no_improve = min(max_no_improve * 2, 50)  # Increase patience

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
