"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with CMA-ES for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = 0
        max_calls = self.budget

        best_x = np.zeros(self.dim)
        best_fx = float('inf')

        def counted_func(x):
            nonlocal calls
            if calls >= max_calls:
                raise ValueError("Budget exceeded")
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Phase 1: Multi-start L-BFGS-B from diverse random points
        # Use 20 random starting points
        num_starts = 20
        for i in range(num_starts):
            if time.time() > deadline - 10.0:
                break
            if calls >= max_calls:
                break

            # Generate random starting point in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Run L-BFGS-B with limited iterations
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 20, 'ftol': 1e-4, 'gtol': 1e-3}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 2: CMA-ES refinement around the best point found
        if calls < max_calls and time.time() < deadline - 5.0:
            # Use remaining budget for CMA-ES
            remaining_calls = max_calls - calls
            
            # Initialize CMA-ES parameters
            x_center = best_x.copy()
            sigma = 0.5  # initial step size
            
            # Simple CMA-ES implementation
            N = 4  # population size
            # Covariance matrix starts as identity
            C = np.eye(self.dim)
            # Evolution paths
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # CMA-ES parameters
            mu = N // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
            
            # Cholesky decomposition for sampling
            def cholesky(C):
                L = np.linalg.cholesky(C)
                return L
            
            L = cholesky(C)
            
            cma_calls = 0
            stagnation_counter = 0
            max_stagnation = 10
            
            while calls + cma_calls < max_calls and time.time() < deadline - 1.0:
                if time.time() > deadline - 2.0:
                    break
                
                # Generate population
                offspring = []
                for i in range(N):
                    z = rng.standard_normal(self.dim)
                    y = x_center + sigma * (L @ z)
                    y_clipped = np.clip(y, -5, 5)
                    offspring.append(y_clipped)
                
                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    if calls >= max_calls:
                        break
                    fy = func(y)
                    calls += 1
                    cma_calls += 1
                    offspring_f.append(fy)
                
                if len(offspring_f) < N:
                    break
                
                # Sort by fitness
                sorted_indices = np.argsort(offspring_f)
                
                # Update best
                best_idx = sorted_indices[0]
                if offspring_f[best_idx] < best_fx:
                    best_fx = offspring_f[best_idx]
                    best_x = offspring[best_idx].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > max_stagnation:
                        break
                
                # Update mean
                y_w = np.average([offspring[i] for i in sorted_indices[:mu]], axis=0, weights=weights)
                y_tilde = (y_w - x_center) / sigma
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_tilde
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2 * cma_calls)) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1)) else 0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_tilde
                
                # Update covariance
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu / mu_eff * np.sum([np.outer(offspring[i] - x_center, offspring[i] - x_center) for i in sorted_indices[:mu]], axis=0) / sigma**2
                
                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                x_center = y_w
                
                # Update Cholesky
                try:
                    L = cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    L = cholesky(C)

        return best_x, best_fx
# EVOLVE-BLOCK-END
