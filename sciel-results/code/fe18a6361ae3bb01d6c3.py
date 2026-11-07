"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(d))  # population size
        mu = lambda_ // 2  # number of parents
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + 4 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        eval_count = 0
        best_x = None
        best_f = float('inf')
        
        def check_time():
            return time.time() < deadline
        
        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f, x_clipped
        
        # Initial evaluation
        f0, x0 = evaluate(x_mean)
        
        # Precompute Cholesky factor
        L = np.linalg.cholesky(C)
        
        # Main CMA-ES loop
        while eval_count < budget and check_time():
            # Generate lambda samples
            samples = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                x_sample = x_mean + sigma * (L @ z)
                samples.append(x_sample)
            
            # Evaluate all samples
            f_samples = []
            x_samples = []
            for x_s in samples:
                if eval_count >= budget or not check_time():
                    break
                f_s, x_s_clipped = evaluate(x_s)
                f_samples.append(f_s)
                x_samples.append(x_s_clipped)
            
            if len(f_samples) < mu:
                break
            
            # Sort by fitness and select top mu
            sorted_idx = np.argsort(f_samples)[:mu]
            x_selected = np.array([x_samples[i] for i in sorted_idx])
            
            # Compute new mean
            x_new_mean = np.sum(weights[:, np.newaxis] * x_selected, axis=0)
            
            # Update p_sigma
            step = x_new_mean - x_mean
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-12:
                h_sigma = 1.0  # Simplified
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (step / sigma)
            
            # Update p_c
            h_c = 1.0  # Simplified
            p_c = (1 - c_c) * p_c + c_c * (step / sigma)
            
            # Update covariance
            C = (1 - c_c - c_1 - c_mu) * C
            C += c_c * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(d))
            C += c_1 * (np.eye(d) - C)
            
            # Rank-mu update
            if c_mu > 0:
                H = np.zeros((d, d))
                for i in range(mu):
                    y_i = (x_selected[i] - x_mean) / sigma
                    H += weights[i] * np.outer(y_i, y_i)
                C += c_mu * (H - C)
            
            # Ensure positive semi-definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma * d_sigma * (step_norm / sigma - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_new_mean
            
            # Update Cholesky factor
            L = np.linalg.cholesky(C)
            
            # Check for convergence
            if sigma < 1e-7:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
