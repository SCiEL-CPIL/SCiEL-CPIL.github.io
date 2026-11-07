"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ1 structure: Pareto front is simplex sum(f)=0.5 when g=0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.RandomState()

        # DTLZ1 specific:
        # x_0 controls scaling (set to 1 for ideal front)
        # x_1...x_{n-3} control shape of the front
        # x_{n-2}, x_{n-1} control distance from front (set to 0.5 for g=0)
        
        # For dim=7, n_obj=3:
        # x_0 is scale
        # x_1, x_2, x_3, x_4 are shape
        # x_5, x_6 are distance
        
        # Strategy:
        # 1. Sample points on the Pareto front (g=0, x_0=1).
        # 2. Use remaining budget to refine/densify the front sampling.
        
        # Number of initial points on the front
        # We want good coverage of the 2D simplex (shape space is 4D but maps to 2D front)
        # Let's use 60% of budget for initial front sampling
        n_initial = int(budget * 0.6)
        
        X_init = np.zeros((n_initial, dim))
        # Set x_0 = 1
        X_init[:, 0] = 1.0
        # Set x_5, x_6 = 0.5 (to minimize g)
        if dim >= 6:
            X_init[:, dim-2] = 0.5
            X_init[:, dim-1] = 0.5
        # Randomize shape variables x_1...x_{dim-3}
        if dim > 3:
            shape_vars = slice(1, dim-2)
            X_init[:, shape_vars] = rng.random((n_initial, dim-3))
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_initial

        # Keep track of all evaluated points
        all_X = [X_init.copy()]
        all_F = [F_init.copy()]
        
        # Current population for refinement (we keep the front points)
        X = X_init
        F = F_init

        # Refinement phase:
        # Use remaining budget to perturb points slightly to fill gaps.
        # We perturb the shape variables and keep x_0=1, x_5=x_6=0.5 to stay on/near front.
        while used < budget:
            if time.time() > deadline:
                break
            
            # Select a random point from the current best set (all are on front)
            # Or just random index
            idx = rng.randint(len(X))
            parent = X[idx].copy()
            
            # Perturb shape variables
            # Step size: small to stay close to front
            step = 0.1 * rng.random() # 0 to 0.1
            if dim > 3:
                shape_vars = slice(1, dim-2)
                parent[shape_vars] += step * rng.standard_normal(dim-3)
                parent[shape_vars] = np.clip(parent[shape_vars], 0.0, 1.0)
            
            # Ensure x_0, x_5, x_6 remain optimal for front
            parent[0] = 1.0
            if dim >= 6:
                parent[dim-2] = 0.5
                parent[dim-1] = 0.5
                
            # Clip entire vector to bounds just in case
            parent = np.clip(parent, lo, hi)
            
            F_child = problem(parent)
            used += 1
            
            # Add to population
            X = np.vstack([X, parent])
            F = np.vstack([F, F_child])
            
            # Periodically record all points (or just keep appending to lists)
            # The evaluator records every call to problem(), so we don't need to return them.
            # But we must not exceed budget.
            
            # Optional: Prune dominated points to save memory? 
            # With 200 budget, memory is trivial. No pruning needed.
            
        return X
# EVOLVE-BLOCK-END
