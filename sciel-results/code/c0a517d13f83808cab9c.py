"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-objective optimizer using Multi-Start Weighted Scalarization with Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s/330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Generate a diverse set of initial points
        # We will use a portion of the budget for initial random sampling
        n_init = min(50, budget // 4)
        if n_init < 10:
            n_init = 10
            
        rng = np.random.default_rng(42)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Track all evaluated points to build the Pareto front
        all_X = X_init.copy()
        all_F = F_init.copy()
        used = n_init

        # 2. Define a function to perform local search on a weighted sum
        def weighted_sum_objective(x, weights):
            # Check deadline occasionally if we were doing many steps, 
            # but minimize is a black box. We rely on maxiter.
            try:
                fx = problem(x)
                return np.dot(weights, fx)
            except Exception:
                return 1e9

        # 3. Multi-Start Local Search
        # We will pick a set of weight vectors to guide the search.
        # Since n_obj=3, we can generate a grid of weight vectors or random ones.
        # We have budget - used evaluations left.
        # Each Nelder-Mead run consumes evaluations. 
        # Let's estimate cost per run. For dim=5, NM might take ~20-50 evals to converge.
        # We can afford roughly (budget - used) / 30 runs.
        
        remaining_budget = budget - used
        if remaining_budget <= 0:
            return all_X

        # Estimate number of local search runs we can afford
        # Be conservative: assume 25 evals per run on average
        n_runs = max(1, remaining_budget // 30)
        
        # Generate weight vectors. 
        # To cover the front well, we want weights that emphasize different objectives.
        # Simple approach: random weights on the simplex.
        # Or structured: for 3 objectives, we can use a grid.
        # Let's use a mix: some extreme points (1,0,0), (0,1,0), (0,0,1) and some mixed.
        
        weight_vectors = []
        # Extreme points
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        
        # Random weights to fill the rest
        while len(weight_vectors) < n_runs:
            w = rng.random(n_obj)
            w /= np.sum(w)
            weight_vectors.append(w)
        
        # Shuffle to avoid bias in early runs
        indices = rng.permutation(n_runs)
        
        for i in indices:
            if used >= budget:
                break
            if time.time() > deadline:
                break
                
            w = weight_vectors[i]
            
            # Select a starting point for this local search
            # Choose the point that best aligns with this weight vector (lowest weighted sum)
            # among the current known points
            weighted_sums = all_F @ w
            start_idx = np.argmin(weighted_sums)
            x0 = all_X[start_idx]
            
            # Run Nelder-Mead
            # maxiter is not exactly eval count, but we can limit it.
            # For dim=5, maxiter=100 is usually plenty for convergence.
            # We don't have a direct eval counter hook in minimize, 
            # so we must be careful. 
            # However, the problem statement says "More than 200 evaluations... invalid".
            # We must count evaluations manually if we want to be safe, 
            # but scipy doesn't expose that easily without a callback.
            # Let's use a callback to count and stop if we exceed budget.
            
            eval_count = [0]
            def callback(xk):
                eval_count[0] += 1
                if used + eval_count[0] >= budget:
                    # Returning False stops the optimization
                    return False
                return None
            
            try:
                res = minimize(
                    lambda x: weighted_sum_objective(x, w), 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4},
                    callback=callback
                )
            except Exception:
                continue
                
            # Update used count
            # Note: minimize calls the function multiple times. 
            # The callback counts function evaluations.
            # However, if the optimization stops early, eval_count is accurate.
            # If it finishes, it's accurate.
            # But wait, does the callback count the initial evaluation? 
            # Yes, usually. 
            # Let's add the count to used.
            used += eval_count[0]
            
            if res.success or eval_count[0] > 0:
                # Add the result to our archive
                all_X = np.vstack([all_X, res.x])
                all_F = np.vstack([all_F, np.array(problem(res.x))])
                
        return all_X
# EVOLVE-BLOCK-END
