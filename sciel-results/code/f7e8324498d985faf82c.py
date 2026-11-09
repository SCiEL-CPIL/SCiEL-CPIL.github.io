"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

CMA-ES approach with adaptive covariance matrix:
  - CMA-ES adapts a multivariate normal to the landscape shape.
  - Path-based covariance updates handle the non-separable structure of f14.
  - Population of 4 for 5D problem balances exploration and exploitation.
  - Initial sigma of 5.0 covers the box quickly.
  - Budget management: use all 500 evaluations efficiently.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14 with hidden affine transform."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, calls_counter, deadline, start_time, dim):
        """Run CMA-ES optimization.
        
        Returns (best_x, best_f).
        """
        # CMA-ES parameters
        n = dim
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n)) if n > 1 else 4
        mu = lam // 2  # number of recombination weights
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()
        
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = (2 + 2 * mueff / (n + 1) ** 2) / ((n + 2) ** 2 + 2 * mueff / (n + 1) ** 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # State
        mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = x0.copy()
        best_f = float(func(x0))
        calls_counter[0] += 1
        
        # Track best
        archive_x = [x0.copy()]
        archive_f = [best_f]
        
        gen = 0
        while calls_counter[0] < budget and time.time() < deadline:
            # Sample population
            pop = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            
            for i in range(lam):
                if calls_counter[0] >= budget or time.time() >= deadline:
                    break
                # Sample from N(mean, sigma^2 * C)
                B = np.linalg.cholesky(C)
                z = np.random.randn(n)
                x_new = mean + sigma * B @ z
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = float(func(x_new))
                calls_counter[0] += 1
                pop[i] = x_new
                pop_f[i] = f_new
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            if calls_counter[0] >= budget or time.time() >= deadline:
                break
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
            
            # Weighted recombination
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update evolution paths
            C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * C_sqrt_inv @ (mean - mean_old) / sigma
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * C_sqrt_inv @ (mean - mean_old) / sigma
            
            # Hessian-free rank-mu and rank-one update
            v = ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            hsig = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - ds) ** (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            C = (1 + c1 * hsig * (np.outer(ps, ps) - C) + cmu * sum(w_i * np.outer(pop[i] - mean_old, pop[i] - mean_old) / sigma ** 2 for i, w_i in enumerate(weights[:mu])) - (c1 * hsig + cmu) * C)
            
            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            # Add small regularization for numerical stability
            C += 1e-12 * np.eye(n)
            
            # Update sigma (CSA)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            gen += 1
        
        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # 10s margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls_counter = [0]
        
        def still_ok():
            return time.time() < deadline and calls_counter[0] < budget
        
        # Initial point: center of the box
        x0 = np.zeros(dim)
        
        # Run CMA-ES with full budget
        # Initial sigma large enough to cover the box
        sigma0 = 5.0
        
        best_x, best_f = self._cma_es(
            func, x0, sigma0, budget, calls_counter, deadline, start, dim
        )
        
        # If we have remaining budget and time, do a local refinement with smaller sigma
        remaining = budget - calls_counter[0]
        if remaining > dim + 2 and still_ok():
            # Try a few more steps with reduced sigma for fine-tuning
            try:
                bx2, bf2 = self._cma_es(
                    func, best_x, sigma0 * 0.1, remaining, calls_counter, deadline, start, dim
                )
                if bf2 < best_f:
                    best_f = bf2
                    best_x = bx2.copy()
            except:
                pass
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        return best_x, float(best_f)
