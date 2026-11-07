"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Memetic optimizer for RE34 using local search on weighted sums."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will record all evaluated points to compute hypervolume at the end
        # But the problem statement says "Every returned objective vector is recorded".
        # The evaluator records the return values of problem(x).
        # We just need to make sure we call problem(x) and don't exceed budget.
        
        used = 0
        evaluated_points = [] # List of (x, f)

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            used += 1
            evaluated_points.append((x, f))
            return f

        # Phase 1: Random initialization
        n_init = 20
        if self.budget < n_init:
            n_init = self.budget
            
        # Use a fixed seed for reproducibility within the call if needed, 
        # but the evaluator seeds np.random. We should use np.random directly or 
        # a derived rng. The prompt says "evaluator calls np.random.seed(seed)".
        # So we can use np.random.
        
        X_init = np.random.uniform(lo, hi, size=(n_init, dim))
        F_init = np.array([eval_point(x) for x in X_init])
        
        # Filter non-dominated solutions
        def is_non_dominated(f, F):
            # f: (n_obj,), F: (N, n_obj)
            # f is non-dominated if no row in F is better in all and better in at least one
            for i in range(len(F)):
                if np.all(F[i] <= f) and np.any(F[i] < f):
                    return False
            return True

        # Keep track of non-dominated set
        nondom_X = []
        nondom_F = []
        for i in range(n_init):
            if is_non_dominated(F_init[i], F_init):
                nondom_X.append(X_init[i])
                nondom_F.append(F_init[i])
        
        if not nondom_X:
            # Fallback if all dominated (unlikely with random)
            nondom_X = [X_init[0]]
            nondom_F = [F_init[0]]

        # Phase 2: Local Search on selected non-dominated points
        # We select a few points and refine them using Nelder-Mead on weighted sums.
        # We need to distribute the remaining budget.
        remaining_budget = self.budget - used
        
        # Select up to 5 points for local search
        n_local = min(5, len(nondom_X))
        
        if n_local > 0 and remaining_budget > 10:
            # Generate weight vectors for local search
            # We want to cover the objective space.
            # Simple strategy: pick n_local points from nondom_X and assign random weights.
            # Or, better: define n_local weight vectors and start local search from 
            # the point in nondom_X that is "closest" to the ideal point in that direction?
            # Simpler: Just pick n_local random points from nondom_X and refine them.
            
            # Let's pick the 'best' points. 
            # We can just take the first n_local non-dominated points.
            # To ensure diversity, let's pick points that are far apart in objective space.
            # For simplicity and robustness, we'll just pick n_local random indices from nondom_X.
            
            indices = np.random.choice(len(nondom_X), size=n_local, replace=False)
            
            # Distribute remaining budget among local searches
            # Reserve some for final random checks? 
            # Let's give ~20-30 evals to each local search.
            evals_per_search = max(10, remaining_budget // n_local)
            
            for idx in indices:
                x_start = nondom_X[idx].copy()
                
                # Define a weighted sum objective to minimize
                # Random weights to explore different parts of the front
                w = np.random.random(n_obj)
                w = w / np.sum(w)
                
                def obj_func(x):
                    # Clip to bounds to ensure validity
                    x_clip = np.clip(x, lo, hi)
                    f = problem(x_clip)
                    nonlocal used
                    # This counts towards the budget!
                    if used >= self.budget:
                        return 0.0 # Stop search
                    used += 1
                    # Record the point
                    evaluated_points.append((x_clip, f))
                    return np.dot(w, f)
                
                # Use Nelder-Mead
                # maxiter controls the number of iterations.
                # Each iteration may require multiple function evaluations.
                # In Nelder-Mead, typically ~n+1 evals per iteration.
                # So maxiter=5 might use ~30 evals for dim=5.
                # We want to use ~evals_per_search evaluations.
                # Let's estimate maxiter = evals_per_search / (dim + 1)
                max_iter = max(1, evals_per_search // (dim + 1))
                
                try:
                    res = minimize(obj_func, x_start, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                except Exception:
                    pass

        # Phase 3: Use any remaining budget for random sampling
        while used < self.budget:
            x_new = np.random.uniform(lo, hi)
            f_new = eval_point(x_new)
            if f_new is not None:
                # Check if it's non-dominated and add to set if so (for completeness, though not strictly needed for scoring if we just record)
                pass

        return evaluated_points
# EVOLVE-BLOCK-END
