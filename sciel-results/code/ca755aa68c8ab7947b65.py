"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Generic multi-objective optimizer using a population-based stochastic search.
    This is a distinct approach family from analytical front construction.
    
    Strategy:
    1. Initialize a random population.
    2. For each budget step, generate new candidates via mutation around the best
       non-dominated points (elite-guided).
    3. Maintain a set of non-dominated solutions found so far.
    4. Return the final set of non-dominated solutions.
    
    This is a standard EMO-style approach, distinct from the current program's
    deterministic analytical front construction.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Parse bounds
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility within the run
        
        # Storage for all evaluated points and their objectives
        X_all = []
        F_all = []
        
        # Non-dominated set
        non_dom_X = []
        non_dom_F = []
        
        def is_dominated(a, b):
            """Check if a dominates b (a is better in all objectives, strictly better in at least one)"""
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(point, objectives, nd_X, nd_F):
            """Check if a new point is non-dominated w.r.t. the current non-dominated set"""
            for i in range(len(nd_F)):
                if is_dominated(objectives, nd_F[i]):
                    return False
            return True
        
        def remove_dominated(new_point, new_obj, nd_X, nd_F):
            """Remove any points in nd_X/nd_F that are dominated by the new point"""
            keep_idx = []
            for i in range(len(nd_F)):
                if not is_dominated(new_obj, nd_F[i]):
                    keep_idx.append(i)
            return np.array([nd_X[i] for i in keep_idx]), np.array([nd_F[i] for i in keep_idx])
        
        # Initial population
        pop_size = min(20, max_evals)
        for _ in range(pop_size):
            if n_evals >= max_evals:
                break
            x = rng.uniform(lows, highs)
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
            n_evals += 1
            
            # Check if non-dominated
            if is_non_dominated(f, f, non_dom_X, non_dom_F):
                non_dom_X.append(x)
                non_dom_F.append(f)
                # Remove any dominated points
                if len(non_dom_F) > 1:
                    non_dom_X, non_dom_F = remove_dominated(x, f, non_dom_X, non_dom_F)
        
        # Main loop
        while n_evals < max_evals:
            # Select a parent from the non-dominated set
            if len(non_dom_F) == 0:
                # Fallback: random
                x = rng.uniform(lows, highs)
            else:
                idx = rng.randint(len(non_dom_F))
                parent = non_dom_X[idx]
                # Mutate
                sigma = 0.1
                x = parent + rng.normal(0, sigma, dim)
                # Clip to bounds
                x = np.clip(x, lows, highs)
            
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
            n_evals += 1
            
            # Update non-dominated set
            if is_non_dominated(f, f, non_dom_X, non_dom_F):
                non_dom_X.append(x)
                non_dom_F.append(f)
                if len(non_dom_F) > 1:
                    non_dom_X, non_dom_F = remove_dominated(x, f, non_dom_X, non_dom_F)
        
        # Return the non-dominated set
        if len(non_dom_X) == 0:
            return np.array(X_all[:1])
        return np.array(non_dom_X)
# EVOLVE-BLOCK-END
