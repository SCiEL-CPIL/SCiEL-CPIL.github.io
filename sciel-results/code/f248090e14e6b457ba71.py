"""CMA-ES + Nelder-Mead hybrid optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a hybrid CMA-ES and Nelder-Mead strategy.
    1. CMA-ES for global search and handling rotations/shifts.
    2. Nelder-Mead for local refinement and fast convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def _count(self):
        self.calls += 1
        return self.calls

    def _evaluate(self, func, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self._count()
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: CMA-ES for global search
        # Use 75% of budget for CMA-ES
        cma_budget = int(0.75 * self.budget)
        
        # Initial parameters for CMA-ES
        x0 = rng.uniform(-5, 5, self.dim)
        sigma0 = 2.0  # Initial step size
        
        # CMA-ES implementation
        def cma_es(func, x0, sigma0, budget, dim, deadline):
            n = dim
            mu = max(1, n // 2)  # Number of parents
            lambda_ = 2 * mu  # Population size
            cc = (4 + n / mu) / (n + 4 + 2 / mu)  # Cumulation parameter
            cs = (mu + 2 - cc) / (n + mu + 5)  # Evolution path length for sigma
            c1 = 2 / ((n + 1.3) ** 2 + mu)  # Learning rate for covariance
            cmu = min(1 - c1, 2 * ((mu + 2 / mu - 2) / ((n + 2) ** 2 + mu)))  # Learning rate for covariance from recombination
            
            x_mean = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)  # Evolution path for sigma
            pc = np.zeros(n)  # Cumulation path
            C = np.eye(n)  # Covariance matrix
            C_inv = np.eye(n)  # Inverse covariance matrix
            sqrt_C = np.eye(n)  # Square root of covariance
            inv_sqrt_C = np.eye(n)  # Inverse square root
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            best_x = x_mean.copy()
            best_f = float(func(best_x))
            calls = 1
            
            # Track best
            if best_f < self.best_f:
                self.best_f = best_f
                self.best_x = best_x.copy()
            
            while calls < budget and time.time() < deadline:
                # Generate population
                W = np.zeros((lambda_, n))
                ws = np.zeros(lambda_)
                # Weighting
                for i in range(mu):
                    ws[i] = np.log(mu + 0.5) - np.log(i + 1)
                ws = ws / ws.sum()
                mueff = 1 / np.sum(ws ** 2)
                
                # Update inverse sqrt
                # Simple approximation: recompute every 10 iterations
                if calls % 10 == 1:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                
                for i in range(lambda_):
                    zi = rng.standard_normal(n)
                    xi = x_mean + sigma * (sqrt_C @ zi)
                    xi = np.clip(xi, -5, 5)
                    fi = float(func(xi))
                    calls += 1
                    self.calls += 1
                    if fi < self.best_f:
                        self.best_f = fi
                        self.best_x = xi.copy()
                    W[i] = xi
                    ws[i] = fi
                
                # Sort by fitness
                order = np.argsort(ws)
                W = W[order]
                ws_sorted = ws[order]
                
                # Update mean
                x_new = np.sum(ws[:mu, None] * W[:mu], axis=0)
                
                # Update evolution path for sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (x_new - x_mean) / sigma)
                
                # Update sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update cumulation path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma
                
                # Update covariance
                delta_h = (1 - hsig) * cc * (2 - cc)
                C = (1 - c1 - cmu) * C + c1 * ((1 + delta_h) * np.outer(pc, pc) - C)
                for i in range(mu):
                    y = inv_sqrt_C @ (W[i] - x_mean)
                    C += cmu * ws[i] * np.outer(y, y)
                
                x_mean = x_new
            
            return x_mean, self.best_f
        
        # Run CMA-ES
        x_cma, f_cma = cma_es(func, x0, sigma0, cma_budget, self.dim, self.deadline)
        
        # Phase 2: Local refinement using Nelder-Mead
        remaining_budget = self.budget - self.calls
        if remaining_budget > 0:
            x0_local = self.best_x.copy()
            
            def objective(x):
                if self.calls >= self.budget or time.time() > self.deadline:
                    return self.best_f
                return self._evaluate(func, x)
            
            try:
                result = minimize(objective, x0_local, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: Final high-precision polish with L-BFGS-B
        remaining_budget = self.budget - self.calls
        if remaining_budget > 10:
            x0_final = self.best_x.copy()
            
            def objective_final(x):
                if self.calls >= self.budget or time.time() > self.deadline:
                    return self.best_f
                return self._evaluate(func, x)
            
            try:
                result = minimize(objective_final, x0_final, method='L-BFGS-B',
                                  bounds=[(-5, 5)] * self.dim,
                                  options={'maxiter': 50, 'gtol': 1e-12})
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass
        
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
