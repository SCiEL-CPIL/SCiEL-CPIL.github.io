"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def func_wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return 1e10
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # We start in the center of the likely minimizer region [-4, 4]^5
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Initial step size (box is size 10, minimizer in size 8)
        
        # Population size
        # Standard rule: 4 + floor(3 * log(dim + 2))
        # For dim=5: 4 + floor(3 * log(7)) ≈ 4 + 3*1.94 ≈ 9.8 -> 9
        # We use 12 for slightly better exploration
        lambda_ = 12
        
        # State variables
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Parameters
        mu = lambda_ // 2  # Number of recombination weights
        # Weights: normalized log-spaced weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control (CSA) parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.3) * (dim + 1.3) + mueff)
        cmu = min(1 - c1, (2 * (mueff - 2 + 1. / mueff)) / ((dim + 2) * (dim + 2) + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
        
        # Precompute Cholesky of C (initially identity)
        C_sqrt = np.eye(dim)
        
        # Main Loop
        while calls < budget and time.time() < deadline:
            # 1. Sample
            ys = np.zeros((lambda_, dim))
            xs = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Standard normal sample
                z = np.random.randn(dim)
                # Transform using C_sqrt and sigma
                y = C_sqrt @ z
                ys[i] = y
                xs[i] = x_mean + sigma * y
            
            # 2. Evaluate
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                fs[i] = func_wrapper(xs[i])
                if calls >= budget or time.time() > deadline:
                    break
            
            if calls >= budget or time.time() > deadline:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            ys_sorted = ys[idx]
            
            # 3. Update Mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * xs_sorted[i]
            
            # 4. Update Covariance Matrix and Step Size
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (x_new_mean - x_mean)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((calls // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chiN)
            hsig = 1.0 if hsig else 0.0
            pc = (1 - cc) * pc + hsig * cc * (x_new_mean - x_mean) / sigma
            
            # Update C
            y_weighted_sum = np.zeros((dim, dim))
            for i in range(mu):
                y_weighted_sum += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            C = (1 - c1 - cmu) * C + c1 * (2 / (1 + cc) - c1 + hsig * cc * (2 - cc)) * C + cmu * y_weighted_sum
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Periodic Cholesky update to maintain numerical stability
            # We do it every 10 iterations or if it fails
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, we might need to restart or regularize
                # For simplicity, we add a small regularization
                C = C + 1e-6 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if all else fails
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)
                    sigma = sigma0
                    x_mean = x0
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    break # Restart logic could be more sophisticated, but this is a safe exit

            # Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for stagnation or divergence
            if not np.isfinite(sigma) or sigma > 100:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
