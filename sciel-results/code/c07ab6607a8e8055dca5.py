"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_iter, rng, calls_counter):
        n = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        lambda_ = min(lambda_, self.budget - calls_counter[0])
        
        # Weights
        weights = np.log(np.arange(1, lambda_ // 2 + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (mu_eff - 2) - 1) / ((n + 2) ** 2 + mu_eff))
        
        # State
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Inverse square root of C
        C_sqrt_inv = np.eye(n)
        
        best_x = x.copy()
        best_fx = func(x)
        calls_counter[0] += 1
        
        for gen in range(max_iter):
            if calls_counter[0] >= self.budget:
                break
                
            # Generate population
            Y = []
            F = []
            for i in range(lambda_):
                if calls_counter[0] >= self.budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_counter[0] += 1
                Y.append(y)
                F.append(fy)
                
            if len(Y) < 2:
                break
                
            # Sort
            indices = np.argsort(F)
            Y = [Y[i] for i in indices]
            F = [F[i] for i in indices]
            
            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()
                
            # Update mean
            m = sum(weights[i] * Y[i] for i in range(len(weights)))
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (m - x) / sigma)
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma) ** 2) / c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
            # Update covariance
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1 - c_c) ** 2) * np.eye(n))
            
            # Weighted sum of outer products
            H = np.zeros((n, n))
            for i in range(len(weights)):
                diff = Y[i] - m
                H += weights[i] * np.outer(diff, diff)
            C += c_mu * H
            
            x = m
            
            # Update C_sqrt_inv
            # Eigen decomposition
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-12, None)
            C_sqrt_inv = (vecs * (1 / np.sqrt(vals))) @ vecs.T
            
            # Check for stagnation or convergence
            if sigma < 1e-8:
                break
            if np.max(vals) / np.min(vals) > 1e12:
                break

        return best_x, best_fx

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls_counter = [0]
        
        best_x = np.zeros(n)
        best_fx = np.inf
        
        # IPOP-CMA-ES style restarts
        # Start with a large sigma, reduce it on restart
        sigma = 2.0
        x0 = np.zeros(n)
        
        while calls_counter[0] < budget:
            # Determine remaining budget
            remaining = budget - calls_counter[0]
            if remaining < 10:
                break
            
            # Run CMA-ES with current settings
            x_opt, f_opt = self._cma_es(func, x0, sigma, remaining, rng, calls_counter)
            
            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt.copy()
                
            # Check if we should restart
            # If we used a significant portion of the budget and didn't converge to a very small sigma
            # Or simply restart with smaller sigma from best point
            if calls_counter[0] >= budget:
                break
                
            # Restart logic:
            # If the last run consumed most of the remaining budget, we might want to restart
            # Here we use a simple heuristic: restart with smaller sigma from best point
            # This helps escape local minima in Lunacek
            
            # Reduce sigma for next restart
            sigma *= 0.5
            x0 = best_x.copy()
            
            # If sigma is too small, break
            if sigma < 0.01:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
