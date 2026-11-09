"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample a small number of points to seed the CMA-ES initial population.
        # We use a standard CMA-ES initialization strategy:
        # x0 = 0 (or a random point), sigma0 = 2.0 (heuristic for [-5,5]^5)
        
        n_init = 10
        rng = np.random.default_rng(42)
        
        # Evaluate a few random points to get a rough sense of the landscape
        # and potentially find a better starting point than 0.
        # Since f11 is rotated, 0 might be far from the optimum.
        for _ in range(n_init):
            if time.time() > deadline:
                break
            p = rng.uniform(-5, 5, self.dim)
            wrapped_func(p)

        # Phase 2: CMA-ES Search
        # We implement a simplified CMA-ES or use scipy's if available and fast.
        # However, scipy's cma_es is not in standard scipy.optimize.minimize.
        # We will implement a basic CMA-ES loop.
        
        # CMA-ES Parameters
        # Lambda = 4 + floor(3 * log(dim))
        lam = int(4 + 3 * np.log(self.dim))
        # Sigma start: heuristic based on domain size
        sigma = 2.0 
        x_mean = self.best_x.copy() if self.best_x is not None else np.zeros(self.dim)
        
        # Covariance matrix C (identity initially)
        C = np.eye(self.dim)
        # Evolution path for covariance
        p_c = np.zeros(self.dim)
        # Evolution path for step size
        p_sigma = np.zeros(self.dim)
        
        # Update rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1) - 1)) + c_sigma
        
        # Precompute for efficiency
        B, D = np.linalg.eigh(C)
        sqrt_inv_C = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
        sqrt_C = B @ np.sqrt(D) @ B.T
        
        def update_state(x_mean, sigma, C, p_c, p_sigma, best_x, best_f):
            # This is a simplified update. For robustness, we might just use
            # scipy.optimize.minimize with method='Nelder-Mead' in a loop or
            # a simpler (1+1)-ES if CMA-ES is too complex to implement correctly
            # in this constrained time.
            # However, CMA-ES is the gold standard for BBOB.
            # Let's try to use a library if possible, or a very robust simple ES.
            # Given the constraint of standard library + numpy + scipy, and no
            # guarantee of a CMA-ES library, I will implement a robust (mu+lambda)-ES
            # with adaptive step size, which is simpler and often sufficient for 5D.
            
            pass

        # Alternative: Use Nelder-Mead with multiple restarts and a large initial simplex.
        # NM is robust and fast for low dimensions.
        
        # Let's do a hybrid: 
        # 1. Run NM from the best point found in Phase 1.
        # 2. If budget remains, run NM from other good points.
        
        # To maximize AOCC, we need fast initial progress.
        # NM with a large initial simplex can provide good global search in 5D.
        
        # Define NM wrapper
        def nm_func(x):
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Run NM from the best point
        if self.best_x is not None and self.call_count < self.budget:
            x0 = self.best_x.copy()
            
            # Initial simplex size
            # If we start at 0, simplex should be large.
            # If we start at a random point, it can be smaller.
            # Let's use a relative size.
            x0_scaled = x0.copy()
            
            try:
                # Use strict tolerances to exploit remaining budget
                # maxiter is limited by budget implicitly via exceptions
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000, 'adaptive': True})
            except ValueError:
                pass
            except Exception:
                pass

        # If we have budget left, do a few more restarts from random points
        # to ensure we haven't missed a better basin.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 50:
            # Use a fraction of remaining budget for restarts
            n_restarts = min(5, remaining_budget // 50)
            for _ in range(n_restarts):
                if self.call_count >= self.budget - 10: # Keep some for final polish
                    break
                if time.time() > deadline - 5:
                    break
                
                x0 = rng.uniform(-5, 5, self.dim)
                try:
                    res = minimize(nm_func, x0, method='Nelder-Mead', 
                                 options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000, 'adaptive': True})
                except ValueError:
                    break
                except Exception:
                    break

        # Final polish with tighter tolerance
        if self.best_x is not None and self.call_count < self.budget:
            x0 = self.best_x.copy()
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                             options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000, 'adaptive': True})
            except ValueError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
