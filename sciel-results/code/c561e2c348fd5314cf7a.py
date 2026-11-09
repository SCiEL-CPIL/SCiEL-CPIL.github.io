"""Two-phase optimizer: Coarse Grid Search followed by Powell's Method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Grid search followed by Powell's method local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        bound = self.bound

        # Global tracking
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -bound, bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Coarse Grid Search ---
        # Use a 3-point grid per dimension: -3, 0, 3 (or similar spread)
        # 3^5 = 243 points. This is < 500.
        grid_points = []
        levels = [-3.0, 0.0, 3.0]
        
        # Generate all combinations
        import itertools
        for combo in itertools.product(levels, repeat=dim):
            grid_points.append(np.array(combo))
        
        # Evaluate grid points
        grid_results = []
        for p in grid_points:
            f = eval_func(p)
            grid_results.append((f, p))
            
        # Sort by function value
        grid_results.sort(key=lambda x: x[0])
        
        # Select top K candidates for local refinement
        # If we have budget left, use it.
        # 243 calls used. 257 left.
        # Let's pick top 3 candidates.
        K = 3
        candidates = [grid_results[i][1] for i in range(min(K, len(grid_results)))]
        
        # --- Phase 2: Powell's Method ---
        # Implement a simplified Powell's method
        def powell(x0, budget_left):
            nonlocal calls, best_x, best_f, deadline
            
            n = dim
            x = x0.copy()
            f = eval_func(x)
            
            # Initial directions: standard basis
            directions = [np.eye(n)[i] for i in range(n)]
            
            # Powell's method loop
            # We will iterate until budget or convergence
            # Each iteration uses n evaluations for line searches + 1 for new direction + 1 for reflection check
            # Cost per full iteration ~ 2n + 1 evaluations. For n=5, ~11 evals.
            # With ~85 budget per candidate, we can do ~7-8 iterations.
            
            max_iter = budget_left // (2 * n + 1) + 2
            
            for it in range(max_iter):
                if calls >= budget_left or time.time() > deadline - 1.0:
                    break
                    
                x_old = x.copy()
                f_old = f
                
                # Line search along each direction
                for i in range(n):
                    if calls >= budget_left or time.time() > deadline - 1.0:
                        break
                    dir_i = directions[i]
                    # Find best step along dir_i
                    # Use a simple ternary search or golden section?
                    # Given the smoothness, a few steps might suffice.
                    # Let's try a small set of steps: -1, -0.5, 0, 0.5, 1 scaled by some factor?
                    # Better: Use a local line search.
                    
                    # Simple line search: try a few steps
                    # We need to determine a step size. Let's use adaptive or fixed.
                    # Fixed step size of 1.0 might be too large or small.
                    # Let's try to find a minimum along the line.
                    
                    # Golden section search for line minimization
                    # But that requires many evals.
                    # Alternative: Since we are in a local basin, a small step might be enough.
                    # Let's use a fixed set of candidate steps: [-2, -1, -0.5, 0.5, 1, 2]
                    # And pick the best.
                    
                    best_step = 0.0
                    best_f_line = f
                    for step in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
                        x_trial = x + step * dir_i
                        f_trial = eval_func(x_trial)
                        if f_trial < best_f_line:
                            best_f_line = f_trial
                            best_step = step
                            
                    x = x + best_step * dir_i
                    f = best_f_line
                    
                # Check for convergence
                if np.linalg.norm(x - x_old) < 1e-6:
                    break
                
                # Update direction
                # If the move was significant, replace the worst direction with the new one
                # Calculate the total move
                move = x - x_old
                if np.linalg.norm(move) > 1e-8:
                    # Determine which direction to replace
                    # The one that gave the largest increase in function value in the previous cycle?
                    # Standard Powell replaces the direction with the largest cost.
                    # We didn't track costs per direction in this simplified version.
                    # Let's just replace the last direction or use a heuristic.
                    # A common simplification: replace the direction that was used least effectively.
                    # For simplicity, let's replace the direction that is most collinear with the move?
                    # Or just cycle through directions?
                    # Let's replace the direction that is most orthogonal to the move? No, that's bad.
                    # Let's replace the direction that contributed least to the reduction.
                    # Since we didn't track that, let's just replace the last direction.
                    directions[-1] = move / np.linalg.norm(move)
                    # Normalize
                    # Ensure it's not zero
                    if np.linalg.norm(directions[-1]) < 1e-10:
                         directions[-1] = np.eye(n)[0]
                         
            return x, f

        # Run Powell's on each candidate
        budget_per_candidate = (budget - calls) // max(1, len(candidates))
        
        for cand in candidates:
            if calls >= budget or time.time() > deadline - 1.0:
                break
            x_res, f_res = powell(cand, budget_per_candidate)
            # The eval_func inside powell already updates global best_x, best_f
        
        return best_x, best_f
# EVOLVE-BLOCK-END
