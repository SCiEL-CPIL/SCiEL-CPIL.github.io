"""Direct-structure optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses the known structure of the function: f(x) = (Griewank(x) - 1) * (Rosenbrock(x) + 450).
Since Rosenbrock is large and positive, minimizing f is dominated by minimizing Rosenbrock,
which has a known global minimum at x* = 1 in each dimension. We exploit this structure
directly with a CMA-ES + Nelder-Mead hybrid. The hidden shift/rotation means the true
minimizer is somewhere in [-4,4]^5, but the landscape is still shaped by the
Griewank-Rosenbrock interaction.

Strategy:
1. Focused multi-start around the Rosenbrock minimum region (x=1)
2. CMA-ES for robust local optimization handling correlated dimensions
3. Nelder-Mead for fast convergence in smooth basins
4. L-BFGS-B final polish for high precision
"""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        
        # Evaluate a candidate and track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initialize with diverse points informed by the structure
        # The Rosenbrock part wants x=1, Griewank wants x=0
        # The product structure means the minimum is a compromise
        rng = np.random.default_rng(42)
        
        # Structured initial points - focus on Rosenbrock minimum region
        init_points = []
        # Point at the known Rosenbrock minimum (strongest prior)
        init_points.append(np.ones(dim))
        # Nearby points around x=1
        for scale in [0.1, 0.2, 0.3, 0.5]:
            for _ in range(2):
                init_points.append(np.ones(dim) + scale * rng.standard_normal(dim))
        # Point at origin (Griewank minimum)
        init_points.append(np.zeros(dim))
        # Points in between
        for t in [0.3, 0.5, 0.7, 0.9]:
            init_points.append(t * np.ones(dim))
        # Random points in a focused region around the expected minimum
        for _ in range(6):
            init_points.append(rng.uniform(-1, 3, dim))
        # A few fully random for exploration
        for _ in range(3):
            init_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate initial points
        for p in init_points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)
        
        # If we've used too many calls, return best
        if call_count >= budget or time.time() >= deadline:
            return best_x
        
        # CMA-ES implementation for robust local optimization
        # Handles correlated dimensions better than coordinate descent
        def cma_es(x0, sigma0, max_iter=30, pop_size=None):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            if pop_size is None:
                pop_size = max(4 + int(3 * np.log(dim)), 8)
            
            x = np.clip(x0.copy(), -5, 5)
            sigma = sigma0
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            
            # Precompute constants
            mu = pop_size // 2
            w = np.zeros(mu)
            w_sum = 0.0
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum = w.sum()
            w = w / w_sum
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            def update_cov(C, C_inv_sqrt, w, x_pop, x_mean, sigma, c_1, c_mu, chi_n):
                # Update covariance matrix
                C_new = (1 - c_1 - c_mu) * C
                for i in range(mu):
                    y_i = (x_pop[i] - x_mean) / sigma
                    C_new += w[i] * np.outer(y_i, y_i)
                # Ensure symmetry and positive definite
                C_new = (C_new + C_new.T) / 2
                # Regularize
                C_new += 1e-10 * np.eye(dim)
                return C_new
            
            for it in range(max_iter):
                if call_count >= budget or time.time() >= deadline:
                    break
                
                # Generate population
                x_pop = np.zeros((pop_size, dim))
                f_pop = np.zeros(pop_size)
                
                for i in range(pop_size):
                    if call_count >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_new = x + sigma * (C_inv_sqrt @ z)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = float(func(x_new))
                    call_count += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                    x_pop[i] = x_new
                    f_pop[i] = f_new
                
                if call_count >= budget or time.time() >= deadline:
                    break
                
                # Sort by fitness
                idx = np.argsort(f_pop)
                x_pop = x_pop[idx]
                f_pop = f_pop[idx]
                
                # Compute weighted mean
                x_mean = np.zeros(dim)
                for i in range(mu):
                    x_mean += w[i] * x_pop[i]
                
                # Update step size
                y_mean = (x_mean - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                x = x_mean
                
                # Update covariance
                C = update_cov(C, C_inv_sqrt, w, x_pop[:mu], x_mean, sigma, c_1, c_mu, chi_n)
                
                # Recompute inverse square root
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                # Reset if sigma gets too small
                if sigma < 1e-10:
                    break
            
            return
        
        # Run CMA-ES from the best starting point
        if call_count < budget and time.time() < deadline:
            cma_es(best_x.copy(), 0.5, max_iter=25, pop_size=10)
        
        # Nelder-Mead optimization from the best point for fast convergence
        def bounded_nm(x0):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            def penalized(x):
                if call_count >= budget or time.time() >= deadline:
                    return np.inf
                x = np.clip(x, -5, 5)
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            
            try:
                res = minimize(
                    penalized,
                    np.clip(x0, -5, 5),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 0.001, 'fatol': 1e-10},
                )
                if res.fun < best_f:
                    eval_point(res.x)
            except Exception:
                pass
        
        # Run NM from best start
        if call_count < budget and time.time() < deadline:
            bounded_nm(best_x.copy())
        
        # If we still have budget, do a second CMA-ES with smaller step
        if call_count < budget and time.time() < deadline:
            cma_es(best_x.copy(), 0.1, max_iter=20, pop_size=8)
        
        # Final high-precision polish pass with L-BFGS-B
        if call_count < budget and time.time() < deadline:
            def polish_func(x):
                if call_count >= budget or time.time() >= deadline:
                    return best_f
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = np.clip(x, -5, 5).copy()
                return f
            
            try:
                res = minimize(
                    polish_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
                )
                if res.fun < best_f:
                    eval_point(res.x)
            except Exception:
                pass
        
        return best_x
