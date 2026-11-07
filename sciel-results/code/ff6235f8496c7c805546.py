import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        # Strategy: Sobol sampling + Local Nelder-Mead refinement
        # 1. Generate Sobol points for global coverage
        # 2. Evaluate a subset
        # 3. Refine best points using local search

        # Phase 1: Global Sampling with Sobol
        # Use 60% of budget for global sampling
        n_global = int(budget * 0.6)
        
        # Generate Sobol sequence
        # Sobol requires power of 2 for best performance, but works for any n
        sobol = qmc.Sobol(d=dim, scramble=True)
        # We need n_global points
        # Sobol .random(n) returns points in [0, 1]
        # We scale to [lo, hi]
        sobol_points = sobol.random(n_global)
        X_global = lo + (hi - lo) * sobol_points
        
        # Evaluate global points
        F_global = np.array([problem(x) for x in X_global], dtype=np.float64)
        used = n_global
        
        # Combine into a single list of (X, F)
        X_all = X_global.copy()
        F_all = F_global.copy()
        
        # Phase 2: Local Refinement
        # Identify non-dominated points from global set
        non_dom_idx = self.find_non_dominated(F_all)
        
        if len(non_dom_idx) == 0:
            # Fallback if something went wrong, though unlikely
            return X_all

        # Select diverse points from non-dominated set for local refinement
        # We want to refine points that are spread out on the Pareto front
        # Use crowding distance or simply pick a few random ones from the non-dominated set
        # Given the small budget, let's pick up to 5-8 points
        
        max_local_runs = 6
        local_eval_budget = budget - used
        
        # Ensure we don't run out of evals
        if local_eval_budget <= 0:
            return X_all
            
        # Determine number of local runs and evals per run
        # Each Nelder-Mead run typically uses at least dim + 1 initial simplex points + iterations
        # For dim=12, simplex has 13 points.
        # Let's reserve 25 evals per local run
        evals_per_run = 25
        n_runs = min(max_local_runs, local_eval_budget // evals_per_run)
        
        if n_runs == 0:
            return X_all

        # Select starting points for local runs
        # Sort non-dominated points by their "quality" (e.g., sum of objectives) and pick a few
        # Or just pick the first n_runs from the non-dominated set
        start_indices = non_dom_idx[:n_runs]
        
        # For each start point, run Nelder-Mead
        # We use a weighted sum scalarization to guide the local search
        # Different weights target different parts of the Pareto front
        
        # Generate random weights for each local run
        # Weights should be positive
        weights_list = []
        for _ in range(n_runs):
            w = np.random.random(n_obj)
            w = w / np.sum(w)
            weights_list.append(w)
            
        # Shuffle weights to ensure diversity
        np.random.shuffle(weights_list)
        
        for i in range(n_runs):
            if time.time() > deadline:
                break
                
            if used >= budget:
                break
                
            start_x = X_all[start_indices[i]]
            w = weights_list[i]
            
            # Define scalarized objective
            def obj(x):
                # Clip x to bounds to be safe, though Nelder-Mead might step out
                # The problem function clips anyway, but we want to stay in bounds for the optimizer
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                return np.dot(w, f)
                
            # Run Nelder-Mead
            # maxiter controls the number of function evaluations roughly
            # For dim=12, initial simplex is 13 points.
            # Let's set maxiter to be small, e.g., 5-10 iterations
            # Total evals approx 13 + maxiter * (dim+1)
            # With evals_per_run=25, we can afford about 1-2 iterations of Nelder-Mead
            # This is very small. Maybe Nelder-Mead is too expensive for this budget?
            # Alternative: Simple Coordinate Descent or Random Perturbation?
            # Let's try Nelder-Mead with very low maxiter.
            
            try:
                result = minimize(obj, start_x, method='Nelder-Mead', 
                                  options={'maxiter': 5, 'xatol': 1e-6, 'fatol': 1e-6})
                # result.x is the best point found
                # result.nfev is the number of function evaluations used
                x_best = np.clip(result.x, lo, hi)
                f_best = problem(x_best)
                
                # We need to track the actual evals used by minimize
                # minimize does not return evals directly in a simple attribute for all methods, 
                # but result.nfev is available.
                evals_used = result.nfev
                used += evals_used
                
                # Add the result to our collection
                X_all = np.vstack([X_all, x_best])
                F_all = np.vstack([F_all, f_best])
                
            except Exception:
                # If minimization fails, just skip
                pass

        return X_all

    def find_non_dominated(self, F):
        """
        Returns indices of non-dominated points.
        """
        n = F.shape[0]
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n):
                if is_dominated[j]:
                    continue
                if self.dominates(F[i], F[j]):
                    is_dominated[j] = True
                elif self.dominates(F[j], F[i]):
                    is_dominated[i] = True
                    break # i is dominated, no need to check further for i
            
        return np.where(~is_dominated)[0]

    def dominates(self, a, b):
        """
        Returns True if a dominates b.
        """
        return np.all(a <= b) and np.any(a < b)
