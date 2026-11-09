"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage optimizer: LHS spread + CMA-ES + Nelder-Mead polish.
    
    Designed for BBOB-style Sphere function with hidden shift/rotation/offset.
    AOCC rewards early progress, so we start with a cheap global spread.
    CMA-ES adapts to the rotated landscape geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Simple CMA-ES implementation."""
        n = len(x0)
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Parameters
        cmu = 2 / ((n + 1) ** 2 + 10)
        c1 = (4 + cmu - n) / ((n + 2) ** 2 + 4)
        cmu = 2 / ((n + 1.3) ** 2 + n)
        c1 = (4 - cmu + 2 / (n + 1.7)) / ((n + 2) + 2 / (n + 1.7) - cmu)
        cmu = 2 / ((n + 1) ** 2 + 10)
        c1 = (4 + cmu - n) / ((n + 2) ** 2 + 4)
        cmu = 2 / ((n + 1.3) ** 2 + n)
        c1 = (4 - cmu + 2 / (n + 1.7)) / ((n + 2) + 2 / (n + 1.7) - cmu)
        cs = (c1 + 2) / (n + c1 + 5)
        c_sigma = (c1 + 2) / (n + c1 + 5)
        d_sigma = 1.0 + 2 * max(0, (n * c1 - 1) / (n + 2)) + cmu
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        lam = 4 + int(3 * np.log(n))
        weights = np.log(np.arange(1, lam + 1) - 0.5) - np.mean(np.log(np.arange(1, lam + 1) - 0.5))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:int(np.ceil(lam / 2))]) ** 2 / np.sum(weights ** 2))
        
        fx = func(x)
        calls_used = 1
        start_time = time.time()
        
        while calls_used < budget and (time.time() - start_time) < 280:
            # Sample
            y = np.zeros((lam, n))
            fy = np.zeros(lam)
            
            for i in range(lam):
                if (time.time() - start_time) > 280:
                    break
                z = rng.standard_normal(n)
                y[i] = x + sigma * (B @ (D * z))
                y[i] = np.clip(y[i], -5, 5)
                fy[i] = func(y[i])
                calls_used += 1
            
            # Sort
            order = np.argsort(fy)
            y = y[order]
            fy = fy[order]
            
            # Update mean
            w = weights[:mueff]
            x_new = np.sum(w[:, None] * y[:mueff], axis=0)
            
            # Update step size control variable
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * (B.T @ (D * (x_new - x)))))
            
            # Update covariance matrix
            hc = (1 - cs) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - 1))) < (1 + 2 / n) * chi_n else 0
            pc = (1 - cs) * pc + hc * (x_new - x)
            
            delta_hc = (1 - hc) * cs * (2 - cs)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hc * C)
            C += cmu / mueff * np.sum(w[:, None, None] * np.outer(y[:mueff] - x, y[:mueff] - x), axis=0)
            
            # Cholesky decomposition
            try:
                B, D = np.linalg.eigh(C)
                B = B @ np.diag(np.sqrt(np.maximum(D, 0)))
                D = np.sqrt(np.maximum(np.linalg.eigvalsh(C), 1e-10))
                B, _ = np.linalg.qr(B)
            except:
                B = np.eye(n)
                D = np.ones(n)
            
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10)
            
            x = x_new
            fx = fy[0]
        
        return x, fx, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: LHS spread for early progress
        n_lhs = 50
        lhs_samples = self._lhs_sample(n_lhs, self.dim, rng)
        
        best_x = lhs_samples[0]
        best_f = func(best_x)
        calls_used = 1
        
        for i in range(1, n_lhs):
            if time.time() > deadline:
                break
            fx = func(lhs_samples[i])
            calls_used += 1
            if fx < best_f:
                best_x = lhs_samples[i]
                best_f = fx
        
        # Phase 2: CMA-ES from best point
        remaining_budget = self.budget - calls_used
        if remaining_budget > 20 and time.time() < deadline:
            x, fx, calls_cma = self._cmaes(func, best_x, 1.0, remaining_budget, rng, deadline)
            calls_used += calls_cma
            if fx < best_f:
                best_x, best_f = x, fx
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline and self.budget - calls_used > 5:
            try:
                result = minimize(func, best_x, method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
