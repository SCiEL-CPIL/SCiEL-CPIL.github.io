"""Optimizer for BBOB-style f12 (Bent cigar) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        # Evaluation counter
        calls = [0]

        def eval(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Phase 1: Global exploration with diverse sampling ---
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Initial diverse sampling to find a good starting basin
        n_init = min(20, budget)
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval(best_x)

        # --- Phase 2: CMA-ES for local optimization ---
        # CMA-ES is well-suited for the Bent Cigar function as it adapts
        # its covariance matrix to the anisotropic landscape.
        
        # Define the objective function for CMA-ES
        def objective(x):
            if calls[0] >= budget or time.time() > deadline:
                return best_f
            return eval(x)

        # Set up CMA-ES parameters
        # Start with a moderate sigma to explore the neighborhood
        sigma0 = 0.5
        
        # Use scipy's CMA-ES if available, otherwise fall back to a custom implementation
        try:
            from cma import CMA
            import cma
            
            # Create CMA-ES options
            opts = {
                'maxiter': 200,
                'bounds': [-5.0, 5.0],
                'verbose': -9,  # Suppress output
                'tolfun': 1e-8,
                'tolfunint': 1e-8,
            }
            
            # Run CMA-ES
            es = CMA(best_x, sigma0, opts)
            
            while not es.stop() and calls[0] < budget - 20 and time.time() < deadline:
                solutions = es.ask()
                for sol in solutions:
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    f_val = eval(sol)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = sol.copy()
                    es.tell(sol, f_val)
                
                # Update best
                if es.result.xbest is not None and es.result.fbest < best_f:
                    best_f = es.result.fbest
                    best_x = es.result.xbest.copy()
                    
        except ImportError:
            # Fallback: Use Nelder-Mead if cma is not available
            result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()

        # --- Phase 3: Fine-tuning with (1+1)-ES on remaining budget ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            sigma_es = 0.05
            x = best_x.copy()
            fx = best_f
            
            # Adaptive step size for ES
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.1
                else:
                    sigma_es *= 0.95
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
