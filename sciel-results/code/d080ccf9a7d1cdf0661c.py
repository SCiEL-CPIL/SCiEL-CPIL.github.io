"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Multi-Start Nelder-Mead on scalarized objectives."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. 
        # We aim to finish well before that to ensure validity.
        deadline = start_time + 290 
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate a diverse set of weight vectors for scalarization.
        # We want to cover the objective space well.
        # For 3 objectives, we can use a set of fixed, well-distributed weights
        # plus some random ones to ensure coverage.
        
        # Strategy: Use a mix of "extreme" weights (focusing on one objective) 
        # and "balanced" weights (focusing on the trade-off surface).
        
        # Fixed well-distributed weights on the simplex for 3 objectives
        # Vertices: (1,0,0), (0,1,0), (0,0,1)
        # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # Center: (1/3, 1/3, 1/3)
        # Additional points to fill gaps
        base_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3.0, 1/3.0, 1/3.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.1, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
            [0.5, 0.2, 0.3],
            [0.3, 0.5, 0.2],
            [0.3, 0.2, 0.5],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
        ]
        
        # Add some random weights to ensure robustness
        n_random_weights = 10
        random_weights = rng.dirichlet(np.ones(self.n_obj), size=n_random_weights)
        
        all_weights = np.vstack([np.array(base_weights), random_weights])
        
        # Budget allocation:
        # Total budget: 200
        # We will perform multiple local optimizations.
        # Each Nelder-Mead run on 5D typically uses ~30-50 function evaluations to converge reasonably.
        # Let's allocate budget to ~5-7 runs.
        
        # To maximize hypervolume, we want points spread across the Pareto front.
        # We will pick a subset of weights that are likely to produce distinct Pareto points.
        
        # Select a diverse subset of weights to run.
        # Let's run 6 local optimizations.
        n_runs = 6
        # Pick evenly spaced indices from the weight list to ensure diversity
        indices = np.linspace(0, len(all_weights)-1, n_runs, dtype=int)
        selected_weights = all_weights[indices]
        
        evals_used = 0
        max_evals_per_run = self.budget // n_runs # ~33 evals per run
        
        results = []
        
        for w in selected_weights:
            if time.time() > deadline:
                break
            
            # Define scalarized objective
            def scalar_obj(x):
                nonlocal evals_used
                if evals_used >= self.budget:
                    return 0.0
                fx = problem(x)
                evals_used += 1
                return np.dot(w, fx)
            
            # Initial point: Random within bounds
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            # Nelder-Mead is derivative-free and robust for small dimensions.
            # We use 'Nelder-Mead' method.
            # Options:
            # maxiter: limit iterations to control time/evals
            # xatol, fatol: tolerance for convergence
            
            # We need to be careful with maxfev to not exceed budget.
            # Nelder-Mead in scipy doesn't have a direct maxfev, but maxiter can help.
            # However, we can wrap the function to count and stop.
            
            # Let's use a wrapper that stops if budget is exceeded or time is up.
            def wrapped_scalar_obj(x):
                if time.time() > deadline or evals_used >= self.budget:
                    # Return a large value to discourage further search in this direction
                    # But we must ensure we don't crash.
                    return 1e10 
                return scalar_obj(x)
            
            try:
                res = minimize(wrapped_scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_evals_per_run * 2, 'xatol': 1e-4, 'fatol': 1e-4})
                
                # Record the best point found by this local run
                # res.x is the best x found
                # We need to evaluate it to get the objective values to record
                # BUT, if it was already evaluated during the search, we might have it.
                # However, the problem statement says "Every returned objective vector is recorded".
                # The `problem` function returns the vector. My `scalar_obj` calls `problem` and returns scalar.
                # The evaluator records the output of `problem(x)`.
                # So every call to `problem` inside `scalar_obj` is recorded.
                # The final `res.x` might not have its vector recorded if it was an internal state 
                # or if we didn't explicitly call `problem(res.x)` at the end.
                # To be safe and ensure the best point of the local search is in the recorded set,
                # we should explicitly evaluate `res.x` if we haven't already or to ensure it's the final state.
                # However, if we call `problem(res.x)` again, it adds 1 eval.
                # If evals_used < budget, we do it.
                
                if evals_used < self.budget and time.time() < deadline:
                    f_final = problem(res.x)
                    evals_used += 1
                    # We don't strictly need to store it in a list for return, 
                    # because the evaluator records all calls to `problem`.
                    # But we return X (list of points) in the seed program. 
                    # The task says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
                    # So we just need to make sure we call `problem` enough times with good points.
                    
            except Exception:
                pass
        
        # If we have remaining budget, we can do a few more random restarts with local search
        # or just random sampling to fill in gaps.
        # But Nelder-Mead is efficient.
        
        # Return dummy, as it is ignored.
        return []

# EVOLVE-BLOCK-END
