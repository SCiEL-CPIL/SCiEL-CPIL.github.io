import numpy as np
import scipy.optimize
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        call_count = [0]
        best_f = float('inf')
        best_x = None
        bounds = [(-5.0, 5.0)] * dim
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget or time.time() > deadline:
                return 1e10
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, lower, upper)
            
            val = func(x)
            if not np.isfinite(val):
                val = 1e10
            
            call_count[0] += 1
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        def polish(x_start, maxfev=20):
            nonlocal best_f, best_x
            if call_count[0] >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func, 
                    x_start.copy(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'maxfev': maxfev,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
                # wrapped_func updates best_f/best_x internally
            except Exception:
                pass

        # Phase 1: Coarse Global Search using Sobol + Random
        # We allocate a small fraction to initial exploration to find promising basins
        n_sobol = 32
        n_random = 18
        n_initial = n_sobol + n_random
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol)
        sobol_pts = sobol_pts * 10.0 - 5.0
        
        # Generate random points
        random_pts = rng.uniform(-5, 5, (n_random, dim))
        
        # Add center point
        center_pt = np.zeros(dim)
        
        all_starts = np.vstack([sobol_pts, random_pts, center_pt.reshape(1, -1)])
        
        # Evaluate initial points
        for pt in all_starts:
            if call_count[0] >= budget or time.time() > deadline:
                break
            wrapped_func(pt)
            
        if best_x is None:
            return np.zeros(dim), best_f

        # Sort evaluated points by function value to get top candidates
        # We need to keep track of (f, x) pairs. Since wrapped_func doesn't return history, 
        # we re-evaluate the best ones? No, that wastes budget.
        # Better approach: We will just run CMA-ES from the global best found so far, 
        # and maybe one or two others if budget allows.
        # However, to do multi-start CMA-ES, we need the starting points.
        # Let's assume the global best is the most important start.
        # For multimodal functions, we might miss other basins.
        # Let's run CMA-ES from the global best.
        
        # Phase 2: CMA-ES Global Refinement
        # CMA-ES is good for adapting to the landscape (rotation, scaling).
        # We run it for a portion of the remaining budget.
        
        remaining_budget = budget - call_count[0]
        if remaining_budget > 50:
            # Use CMA-ES from the best point found
            # CMA-ES typically needs ~ (10-20)*dim evaluations to converge
            # dim=5, so ~50-100 evals.
            # We'll give it a good chunk of the budget.
            cmaes_budget = min(150, int(remaining_budget * 0.6))
            
            if cmaes_budget > 10:
                try:
                    # Simple CMA-ES implementation using scipy's minimize with method='CMA-ES'
                    # Note: scipy.optimize.minimize with method='CMA-ES' is available in recent scipy versions.
                    # If not available, we might need a fallback. Let's assume it is.
                    # Actually, scipy.optimize.minimize does NOT have a built-in CMA-ES method in standard versions.
                    # It has 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', etc.
                    # CMA-ES is in scipy.optimize.cma_es? No, it's usually in cma library or manual implementation.
                    # Wait, scipy.optimize.minimize does NOT support CMA-ES directly as a method string.
                    # I must implement a simple CMA-ES or use a different strategy.
                    
                    # Let's stick to a robust strategy: Multi-start L-BFGS-B with adaptive restarts.
                    # Or implement a very simple (1+lambda) evolution strategy / CMA-ES.
                    
                    # Given the constraint "Python standard library, NumPy and SciPy are available",
                    # and knowing scipy.optimize.minimize methods, I should use what's available.
                    # Nelder-Mead is derivative-free and robust, but can be slow.
                    # Powell is derivative-free and robust.
                    
                    # Let's use Powell for the global refinement phase as it handles non-smoothness better than Nelder-Mead sometimes.
                    # Or better: Run L-BFGS-B from multiple diverse points.
                    
                    # Let's identify the top 3 points from our initial evaluation.
                    # We didn't store them explicitly in a list sorted by value.
                    # Let's re-evaluate a few promising points? No, waste.
                    # We only have best_x.
                    
                    # Strategy:
                    # 1. Polish best_x with L-BFGS-B.
                    # 2. If budget remains, do a local search around best_x with larger steps (Powell).
                    # 3. If budget remains, try a few random restarts around best_x.
                    pass
                except Exception:
                    pass
        
        # Phase 2a: Intensive Polish of the Global Best
        if call_count[0] < budget and time.time() < deadline:
            # Give L-BFGS-B a generous budget to converge
            polish(best_x, maxfev=min(50, budget - call_count[0]))
            
        # Phase 2b: Secondary Exploration
        # If we still have budget, let's try to escape local minima.
        # We can use a simple random walk with decreasing step size (Simulated Annealing style)
        # or just random perturbations.
        
        remaining = budget - call_count[0]
        if remaining > 20:
            # Try a few random perturbations around the best point
            # Step size starts large and decreases
            base_step = 2.0
            n_perturbations = min(20, remaining // 2)
            
            for i in range(n_perturbations):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                
                # Decrease step size
                step = base_step * (0.9 ** i)
                
                # Random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                x_try = best_x + step * direction
                x_try = np.clip(x_try, lower, upper)
                
                f_try = wrapped_func(x_try)
                
                if f_try < best_f:
                    # Improved, polish from here
                    if call_count[0] < budget and time.time() < deadline:
                        polish(x_try, maxfev=min(10, budget - call_count[0]))
                
                # Even if not improved, if we have lots of budget, maybe try another direction?
                # The loop handles this.
        
        # Phase 2c: Final Polish
        if call_count[0] < budget and time.time() < deadline and best_x is not None:
            # One last attempt to squeeze out precision
            polish(best_x, maxfev=budget - call_count[0])

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                wrapped_func(best_x)

        return best_x, best_f
