"""CMA-ES based optimizer for BBOB f8 (Rosenbrock)"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        total_calls = 0

        def check_time():
            return time.time() < deadline

        # CMA-ES parameters
        dim = self.dim
        # Population size: 4 + floor(3 * ln(dim)) is standard, but for small dim and tight budget,
        # we can go slightly smaller to allow more generations. 
        # For dim=5, lambda = 4 + 3*ln(5) ~ 8.9 -> 8. 
        # Let's use lambda = 4 + int(3 * np.log(dim))
        lam = 4 + int(3 * np.log(dim))
        
        # Initial step size
        sigma = 1.0
        # Initial mean
        mean = rng.uniform(-2, 2, dim)
        
        # CMA-ES state
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters
        c_sigma = (lam / 2 + 1) / (dim + 2 + lam / 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        mu = lam // 2  # Number of best points used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial evaluation
        fx_mean = func(mean)
        total_calls += 1
        if fx_mean < best_fx:
            best_fx = fx_mean
            best_x = mean.copy()

        # Main loop
        while total_calls + lam <= self.budget and check_time():
            # Generate offspring
            ys = []
            fs = []
            
            for i in range(lam):
                if total_calls >= self.budget or not check_time():
                    break
                # Generate step
                z = rng.standard_normal(dim)
                y = mean + sigma * (Csqrt @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                total_calls += 1
                ys.append(y)
                fs.append(fy)
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if len(ys) < lam:
                break
            
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            y_w = np.sum(weights[:, None] * ys[:mu], axis=0)
            mean = y_w
            
            # Update sigma
            y_sigma = (y_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (total_calls // lam + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update covariance matrix
            if h_sigma:
                rank_one = np.outer(p_c, p_c)
            else:
                rank_one = np.zeros((dim, dim))
                
            rank_mu = np.sum(weights[:, None, None] * (ys[:mu] - mean)[:, None, :] * (ys[:mu] - mean)[:, :, None], axis=0)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Cholesky decomposition
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity and restart
                C = np.eye(dim)
                Csqrt = np.eye(dim)
                mean = best_x.copy()
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                continue
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        return best_x, best_fx
# EVOLVE-BLOCK-END
