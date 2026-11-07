"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def cmaes_single(start_x, sigma_start, max_steps, seed_offset=0):
            """
            Lightweight CMA-ES implementation for low dimensions.
            Returns (best_x, best_f, calls_made).
            """
            nonlocal call_count, best_f, best_x
            dim = self.dim
            
            # CMA-ES parameters
            # Population size lambda
            lam = 4 + int(3 * np.log(dim)) 
            # Learning rates
            cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            cs = (lam / 2 + 1) / (dim + lam / 2 + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + lam)
            cmu = min(1 - c1, 2 * ((lam / 2 - 1) + 1) / ((dim + 2) ** 2 + lam))
            
            # Initialize
            x_mean = start_x.copy()
            sigma = sigma_start
            C = np.eye(dim)
            Cinv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Precompute weights
            # Use standard weights: top half get higher weight
            weights = np.ones(lam)
            # Simple rank-based weights
            mu_eff = lam // 2
            for i in range(mu_eff):
                weights[i] = np.log((mu_eff + 1) / (i + 1)) - np.log(mu_eff / (mu_eff + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Cholesky of C
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_cholesky = np.eye(dim)
            
            current_best_x = x_mean.copy()
            current_best_f = np.inf
            local_calls = 0
            
            # Initial evaluation
            f = make_call(x_mean)
            if f is None:
                return current_best_x, current_best_f, 0
            local_calls += 1
            if np.isfinite(f) and f < current_best_f:
                current_best_f = f
                current_best_x = x_mean.copy()
                if f < best_f:
                    best_f = f
                    best_x = x_mean.copy()
            
            stagnation_count = 0
            improvement_threshold = 1e-8 * max(1.0, abs(current_best_f))
            
            for gen in range(max_steps):
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Sample population
                pop_x = np.zeros((lam, dim))
                pop_f = np.zeros(lam)
                
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = C_cholesky @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    pop_x[i] = x_new
                    fi = make_call(x_new)
                    if fi is None:
                        break
                    local_calls += 1
                    pop_f[i] = fi
                    if np.isfinite(fi) and fi < current_best_f:
                        current_best_f = fi
                        current_best_x = x_new.copy()
                        if fi < best_f:
                            best_f = fi
                            best_x = x_new.copy()
                
                if call_count >= self.budget:
                    break
                
                # Sort population by fitness
                sorted_idx = np.argsort(pop_f)
                sorted_x = pop_x[sorted_idx]
                sorted_f = pop_f[sorted_idx]
                
                # Check for improvement
                if sorted_f[0] < current_best_f - improvement_threshold:
                    stagnation_count = 0
                    current_best_f = sorted_f[0]
                    current_best_x = sorted_x[0].copy()
                    if sorted_f[0] < best_f:
                        best_f = sorted_f[0]
                        best_x = sorted_x[0].copy()
                else:
                    stagnation_count += 1
                    if stagnation_count > 10:
                        break # Restart triggered by stagnation
                
                # Update mean
                w_sum = np.sum(weights[:mu_eff])
                x_mean_new = np.sum(weights[:mu_eff, np.newaxis] * sorted_x[:mu_eff], axis=0) / w_sum
                x_mean_new = np.clip(x_mean_new, -5, 5)
                
                # Update covariance matrix
                diff = sorted_x[:mu_eff] - x_mean
                C_old = C.copy()
                # Update C using rank-mu update
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc) * chi_n^2) * I) + cmu * sum(wi * diff_i * diff_i.T)
                # Simplified update for speed:
                pc = (1 - cc) * pc + cc * np.sqrt(mu_eff) * (x_mean_new - x_mean) / sigma
                chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (20 * dim ** 2))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_mean_new - x_mean)
                
                # Update C
                rank_mu = np.sum(weights[:mu_eff, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]), axis=0)
                C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc) * chi_n ** 2) * np.eye(dim)) + cmu * rank_mu
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(dim)
                
                x_mean = x_mean_new
                
                # Update sigma
                sigma *= np.exp(cs / sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-6, min(sigma, 10.0))
                
                # Update Cholesky
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(dim)
                    C_cholesky = np.linalg.cholesky(C)
                    
                # Check convergence
                if sigma < 1e-6:
                    break

            return current_best_x, current_best_f, local_calls

        # Main loop: Multiple restarts of CMA-ES
        n_restarts = 5
        # Allocate budget dynamically
        # First restart gets more budget to explore
        budgets = [100, 80, 60, 50, 40]
        
        for r in range(n_restarts):
            if call_count >= self.budget:
                break
            if time.time() > deadline - 10.0:
                break
            
            # Determine start point
            if r == 0:
                x_start = np.zeros(self.dim)
                sigma_start = 2.0
            elif r == 1:
                x_start = rng.uniform(-3, 3, self.dim)
                sigma_start = 1.5
            else:
                # Start near previous best if found, else random
                if best_x is not None and r > 1:
                    x_start = best_x + rng.standard_normal(self.dim) * 0.5
                else:
                    x_start = rng.uniform(-4, 4, self.dim)
                sigma_start = 1.0
                
            x_start = np.clip(x_start, -5, 5)
            
            # Run CMA-ES
            # Limit steps to ensure we don't exceed time
            max_steps = 50 # Generations
            cmaes_single(x_start, sigma_start, max_steps, seed_offset=r)
            
            # If we have plenty of budget left and time, we could do another pass
            # But CMA-ES with restarts is usually sufficient for 500 budget in dim 5

        # Final local refinement using Nelder-Mead on the best found
        if best_x is not None and call_count < self.budget - 10:
            if time.time() < deadline - 5.0:
                # Define a wrapper to count calls for NM
                nm_call_count = 0
                def nm_func(x):
                    nonlocal nm_call_count
                    if nm_call_count >= 20: # Limit NM calls
                        return best_f
                    nm_call_count += 1
                    fx = func(x)
                    return fx if np.isfinite(fx) else best_f
                
                # Nelder-Mead is fast for low dim
                try:
                    result = minimize(nm_func, best_x, method='Nelder-Mead', 
                                      options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.success and result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass

        # Fallback
        if best_x is None:
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
