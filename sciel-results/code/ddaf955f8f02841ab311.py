"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for global
exploration and adaptive local refinement. CMA-ES is robust to multimodality
and adapts its search distribution to the problem's landscape structure.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 295.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Best solution tracking
        best_x = None
        best_f = np.inf
        
        def eval_func(x):
            nonlocal best_x, best_f
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # CMA-ES parameters
        # Initial mean: center of the box
        m = np.zeros(d)
        
        # Initial step size: cover a significant portion of the box
        # Box is [-5, 5], so std dev of 2.5 is reasonable for global exploration
        sigma = 2.5
        
        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        if lambda_ < 4:
            lambda_ = 4
        if lambda_ > 15:
            lambda_ = 15  # Cap for efficiency with small budget
        
        # CMA-ES adaptation rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # State variables
        pc = np.zeros(d)
        B = np.eye(d)
        D = 1.0
        C = np.eye(d)
        
        # Evaluate initial point
        fx0 = eval_func(m)
        best_x = m.copy()
        best_f = fx0
        
        evals_used = 1
        remaining = budget - 1
        
        # Expected norm of a standard normal vector in d dims
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d*d))
        
        while remaining > 0 and time.time() - start_time < timeout:
            # Check if we can fit at least one more generation
            if remaining < lambda_:
                # Do (1+1)-ES like steps for the remainder to squeeze out last bits
                # Use adaptive step size based on recent success
                recent_successes = 0
                total_tries = 0
                for _ in range(remaining):
                    if time.time() - start_time > timeout:
                        break
                    y = m + sigma * np.random.randn(d)
                    y = np.clip(y, -5, 5)
                    fy = eval_func(y)
                    evals_used += 1
                    remaining -= 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        m = y.copy()
                        sigma *= 1.1
                        recent_successes += 1
                    else:
                        sigma *= 0.9
                    total_tries += 1
                    # Adapt sigma based on success rate
                    if total_tries >= 5:
                        success_rate = recent_successes / total_tries
                        if success_rate > 0.5:
                            sigma *= 1.2
                        elif success_rate < 0.2:
                            sigma *= 0.8
                        recent_successes = 0
                        total_tries = 0
                break
            
            # Generate population
            X = []
            for i in range(lambda_):
                # Sample from N(m, sigma^2 * C)
                z = np.random.randn(d)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                X.append(y)
            
            # Evaluate
            F = [eval_func(x) for x in X]
            evals_used += lambda_
            remaining -= lambda_
            
            # Update best
            for i in range(lambda_):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort
            idx = np.argsort(F)
            x_sorted = [X[i] for i in idx]
            f_sorted = [F[i] for i in idx]
            
            # Weighted mean
            m_new = sum(weights[i] * x_sorted[i] for i in range(mu))
            
            # Step size control (CSA)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))
            
            # Heuristic to adapt sigma based on pc length
            if np.linalg.norm(pc) / chi_d < (1.4 + 2/(d+1)):
                sigma /= (1 + c_sigma)
            else:
                sigma *= np.exp(c_sigma * (np.linalg.norm(pc) / chi_d - 1))
            
            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update C
            delta_h = 1.0 if np.linalg.norm(pc) / chi_d < (1.4 + 2/(d+1)) else 0.0
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C)
            
            for i in range(mu):
                diff = x_sorted[i] - m_new
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Ensure C is positive definite and symmetric
            C = (C + C.T) / 2
            # Eigendecomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                # Fallback to identity if it fails
                B = np.eye(d)
                D = np.ones(d)
                C = np.eye(d)
            
            m = m_new
        
        return best_x, best_f
