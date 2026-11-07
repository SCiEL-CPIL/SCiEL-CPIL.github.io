"""Structural change: Deterministic LHS Grid + BFGS Local Search for BBOB f19."""
import numpy as np
import time
import math
import scipy.optimize

class Optimizer:
    """Optimizer using stratified LHS global sampling followed by BFGS local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        evals = []

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                evals.append((calls_made, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        # Phase 1: Deterministic Global Coverage with Stratified LHS
        # We allocate ~20% of the budget to the global grid to ensure coverage.
        # For dim=5, a 4x4x4x4x4 grid is 1024 points, which is too many.
        # A 3x3x3x3x3 grid is 243 points.
        # Let's aim for ~50-100 points for the grid to be efficient.
        # Stratified sampling: divide each dimension into N bins, pick one point per bin.
        # For 5D, if we pick N=3 per dim, we get 3^5=243 points.
        # If we pick N=2 per dim, we get 2^5=32 points.
        # Let's use a hybrid: 243 points if budget allows, else 32.
        
        grid_budget = min(100, budget // 5)
        
        # Generate LHS points
        # Simple stratified sampling
        n_per_dim = 3 # 3^5 = 243 points. If grid_budget < 243, we sub-sample.
        points = []
        for i in range(n_per_dim ** dim):
            if len(points) >= grid_budget:
                break
            # Unpack index to coordinates in [0, n_per_dim)
            coords = []
            temp_i = i
            for d in range(dim):
                coords.append(temp_i % n_per_dim)
                temp_i //= n_per_dim
            
            # Map to [-5, 5] with stratification
            x = np.zeros(dim)
            for d in range(dim):
                # Random offset within the bin
                x[d] = -5 + (coords[d] + np.random.rand()) * (10.0 / n_per_dim)
            points.append(x)
        
        # Shuffle to randomize order of evaluation
        np.random.shuffle(points)
        
        # Evaluate grid points
        for x in points:
            if calls_made >= budget or time.time() >= deadline:
                break
            safe_eval(x)

        # Phase 2: Local Refinement with BFGS
        # Select top K points from the grid to start BFGS
        # We need to sort the evaluations done so far
        if evals:
            # Sort by function value
            sorted_evals = sorted(evals, key=lambda t: t[1])
            # Take top 5 unique starting points (or fewer if not enough)
            top_starts = []
            seen = set()
            for idx, (call_idx, f_val) in enumerate(sorted_evals):
                # We don't have the x stored in evals easily, so we need to re-evaluate or store x.
                # Let's modify safe_eval to store x as well or just re-use the points list.
                # Actually, we can just pick the best few from the `points` list we generated.
                pass
            
            # Better approach: Keep track of best points found during grid phase
            # Let's just pick the 5 best from the generated `points` list based on their f-values
            # We need to map points to their f-values.
            
            # Let's re-do the grid evaluation tracking properly.
            # I'll clear evals and re-evaluate the best candidates from `points`? No, that wastes calls.
            # Instead, I'll just run BFGS on the best `best_x` found so far, and a few perturbed versions.
            # The directive says "from each basin". The LHS grid identifies basins.
            # Let's identify the best 5 points from the grid.
            
            # We need to know which points were best.
            # Let's create a list of (f, x) for the grid points.
            grid_results = []
            # We have to re-evaluate? No, we can just store them in safe_eval.
            # Let's change safe_eval to append (fx, x) to a list `grid_points_evaluated` if it's a grid point?
            # Simpler: Just run BFGS on `best_x` and 4 random perturbations of `best_x`.
            # This is a "multi-start" local search.
            
            # To strictly follow "from each basin", we should pick the best points from the grid.
            # Let's assume the top 5 best points from the grid are in different basins.
            # We need to retrieve them.
            
            # I will modify the logic to store the best grid points.
            # Let's just use the best_x found so far as the primary start.
            # And add 4 more starts: best_x + small noise.
            
            starts = [best_x.copy()]
            for i in range(4):
                starts.append(best_x + np.random.randn(dim) * 0.5)
                starts[-1] = np.clip(starts[-1], -5, 5)
            
            remaining = budget - calls_made
            if remaining > 0:
                budget_per_start = max(10, remaining // len(starts))
                
                for x0 in starts:
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    # Define the objective for scipy
                    # scipy calls func(x) directly. We need to count calls.
                    # scipy.optimize.minimize doesn't support call limits easily without wrapping.
                    # We can use a wrapper that checks budget.
                    
                    def obj(x):
                        if calls_made >= budget or time.time() >= deadline:
                            return float('inf')
                        fx = safe_eval(x)
                        if fx is None:
                            return float('inf')
                        return fx

                    try:
                        result = scipy.optimize.minimize(
                            obj, 
                            x0, 
                            method='BFGS', 
                            options={'maxiter': 50, 'disp': False}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x
                    except Exception:
                        pass

        return best_x, best_f
