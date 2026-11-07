"""
Strategy: Multi-Start Nelder-Mead with Sobol Global Sampling.
1. Global Phase: Use a Sobol sequence to uniformly sample the domain. 
   This is more efficient than random or integer lattice for rotated functions.
2. Local Phase: Run Nelder-Mead simplex optimization from the top global candidates.
   Nelder-Mead adapts to local curvature and is effective in low dimensions.
3. Polish: Use a final tight local search if budget allows.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        n = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            try:
                # Ensure input is a valid numpy array of floats
                x_eval = np.asarray(x, dtype=float)
                # Clip to box [-5, 5] as per problem description, though Nelder-Mead might go out
                # The problem says func(x) clips, but we should be safe.
                x_eval = np.clip(x_eval, -5.0, 5.0)
                
                f_val = func(x_eval)
                calls_made += 1
                
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_eval.copy()
                return f_val
            except Exception:
                return np.inf

        # --- Phase 1: Global Sampling with Sobol Sequence ---
        # We want to cover [-5, 5]^n.
        # Sobol sequences are low-discrepancy, meaning they spread out evenly.
        # We will generate 32 points initially to find a good starting point.
        num_global = 32
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        # Generate points in [0, 1)^n
        sobol_points = sampler.generate(num_global)
        # Map to [-5, 5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate global candidates
        global_results = []
        for x in sobol_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f_val = evaluate(x)
            if f_val < np.inf:
                global_results.append((f_val, x))
        
        if not global_results:
            # Fallback if something went wrong
            return np.zeros(n), np.inf

        # Sort by function value
        global_results.sort(key=lambda item: item[0])
        
        # Select top 2 candidates for local optimization
        # This provides robustness against local minima traps
        candidates = [global_results[0][1], global_results[1][1]] if len(global_results) > 1 else [global_results[0][1]]

        # --- Phase 2: Local Optimization with Nelder-Mead ---
        # Nelder-Mead is robust and doesn't require gradients.
        # We allocate the remaining budget among the candidates.
        
        for x0 in candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Define the objective for scipy
            # We use the existing evaluate function to track calls and best
            def objective(x):
                return evaluate(x)
            
            # Options for Nelder-Mead
            # maxiter is not the number of function calls, but iterations.
            # We need to be careful with the budget.
            # A typical Nelder-Mead in 5D might take ~50-100 evals to converge well.
            # Let's cap the iterations to ensure we don't exceed budget.
            # With ~460 calls left, we can afford ~100-200 per restart.
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-8, 
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                # Evaluate the final result explicitly to ensure it's recorded
                # (minimize might stop early without evaluating the absolute best if it converges on x)
                # But our objective tracks best_f, so best_x should be updated.
                # However, res.x might be slightly better than the last evaluated point if the simplex shrinks?
                # No, minimize evaluates vertices. The best vertex is returned in res.x.
                # Let's evaluate res.x to be absolutely sure it's in our history.
                if calls_made < budget and time.time() < deadline:
                    f_final = evaluate(res.x)
                    
            except Exception:
                pass

        # --- Phase 3: Final Polish (Optional) ---
        # If we have significant budget left, do a tiny CMA-ES or coordinate descent?
        # Nelder-Mead usually does a good job. 
        # Let's check if we have > 50 calls left.
        if calls_made < budget - 50 and best_x is not None:
             # Simple local refinement using a small step size random search
             # Or just run another Nelder-Mead from the current best with tighter tolerance?
             # Let's try a few random perturbations to escape potential flat spots.
             sigma = 0.01
             for _ in range(10):
                 if calls_made >= budget or time.time() > deadline:
                     break
                 x_pert = best_x + sigma * np.random.randn(n)
                 f_pert = evaluate(x_pert)
                 if f_pert < best_f:
                     # If improved, run a quick Nelder-Mead from here
                     try:
                         res = minimize(
                             objective, 
                             x_pert, 
                             method='Nelder-Mead', 
                             options={'maxiter': 50, 'xatol': 1e-9, 'fatol': 1e-14}
                         )
                         if calls_made < budget and time.time() < deadline:
                             evaluate(res.x)
                     except Exception:
                         pass
                         break

        return best_x, best_f
