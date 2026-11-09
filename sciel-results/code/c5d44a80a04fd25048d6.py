import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        max_time = 200.0  # Safety margin for 300s limit
        calls_used = 0
        max_calls = self.budget
        
        best_x = None
        best_f = np.inf
        
        def eval_point(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= max_calls:
                return np.inf
            if time.time() - start_time > max_time:
                return np.inf
            val = float(func(x))
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Deterministic Global Search using Halton Sequence
        # Halton sequence provides better coverage than uniform random in low dimensions
        n_global = 250
        if self.dim > 0:
            # Generate Halton sequence
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            points = []
            for i in range(1, n_global + 1):
                p = np.array([halton(i, base) for base in range(2, 2 + self.dim)])
                # Map [0, 1]^d to [-5, 5]^d
                p = p * 10.0 - 5.0
                points.append(p)
            
            for p in points:
                if calls_used >= max_calls or time.time() - start_time > max_time:
                    break
                eval_point(p)

        if best_x is None:
            # Fallback if something went wrong or no calls were made
            best_x = np.zeros(self.dim)
            eval_point(best_x)

        # Phase 2: Local Search using CMA-ES logic
        # We implement a simple CMA-ES-like step or use scipy's Powell/Nelder-Mead
        # The directive asks for CMA-ES. I will implement a minimal CMA-ES.
        
        x_mean = best_x.copy()
        sigma = 2.0 # Initial step size
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        Cs = np.eye(self.dim)
        Cs_inv = np.eye(self.dim)
        
        # CMA-ES parameters for dim=5
        # lambda = 2 * floor(3 + log(10 * dim)) -> for dim 5, log(50)~3.9, floor(6.9)=6, lambda=12? 
        # Standard is often lambda = 4 + floor(3 log n). For n=5, 4+3*1.6=8.8 -> 9 or 10.
        # Let's use lambda = 9.
        mu = self.dim // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        def cma_step():
            nonlocal x_mean, sigma, C, C_inv, Cs, Cs_inv, p_sigma, p_c
            # Generate lambda samples
            lam = 9
            samples = []
            for i in range(lam):
                z = np.random.randn(self.dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(self.dim)
                x = x_mean + sigma * (L @ z)
                samples.append(x)
            
            # Evaluate
            values = []
            for x in samples:
                if calls_used >= max_calls or time.time() - start_time > max_time:
                    return
                val = eval_point(x)
                values.append(val)
            
            if len(values) < lam:
                return
                
            # Sort
            indices = np.argsort(values)
            sorted_x = [samples[i] for i in indices]
            sorted_v = [values[i] for i in indices]
            
            # Update mean
            x_new_mean = np.zeros(self.dim)
            for i in range(mu):
                x_new_mean += weights[i] * sorted_x[i]
            
            # Update sigma
            x_sigma = (x_new_mean - x_mean) / (sigma * np.sqrt(mu_eff))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * x_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                z_i = (sorted_x[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(z_i, z_i)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definite
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(self.dim)
            
            x_mean = x_new_mean

        # Run CMA-ES until budget/time
        while calls_used < max_calls and time.time() - start_time < max_time:
            cma_step()
            # Check for convergence or step size too small
            if sigma < 1e-6:
                break

        return best_f
