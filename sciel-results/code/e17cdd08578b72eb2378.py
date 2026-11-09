import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        x_best = None
        f_best = np.inf
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        
        def check_deadline():
            return used_calls >= total_budget or time.time() > deadline
        
        # Phase 1: Coarse grid search to find the basin of attraction quickly
        # For f6 (Attractive Sector), the basin is large, so a coarse grid works well
        # Use a 3^5 = 243 point grid, but we only have 500 budget
        # Let's use a smarter approach: evaluate points on a stratified grid
        
        # Create a grid of candidate starting points
        # We'll use a mixture of:
        # 1. Origin and its neighbors
        # 2. Points along each axis
        # 3. Random diverse points
        
        candidate_points = []
        
        # Origin
        candidate_points.append(np.zeros(dim))
        
        # Points along each axis at various distances
        for d in range(dim):
            for dist in [0.5, 1.0, 2.0, 3.0]:
                for sign in [1, -1]:
                    p = np.zeros(dim)
                    p[d] = sign * dist
                    candidate_points.append(p)
        
        # Points in 2D planes
        for d1 in range(dim):
            for d2 in range(dim):
                if d1 >= d2:
                    continue
                for dist in [1.0, 2.0]:
                    for sign1 in [1, -1]:
                        for sign2 in [1, -1]:
                            p = np.zeros(dim)
                            p[d1] = sign1 * dist
                            p[d2] = sign2 * dist
                            candidate_points.append(p)
        
        # Random diverse points
        n_random = 50
        for _ in range(n_random):
            candidate_points.append(rng.uniform(-4, 4, dim))
        
        # Shuffle to get diversity early
        np.random.shuffle(candidate_points)
        
        # Evaluate candidate points until we have a good starting point or run out of budget
        # We want to spend at most ~40% of budget on exploration
        explore_budget = min(200, total_budget // 3)
        
        for i, x in enumerate(candidate_points):
            if used_calls >= explore_budget or check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            used_calls += 1
            if f < f_best:
                f_best = f
                x_best = x_clipped.copy()
        
        # If we haven't found anything, use origin
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = func(x_best)
            used_calls += 1
        
        # Phase 2: Local (1+1)-ES refinement from the best point
        # The attractive sector has a large basin, so local search should work well
        
        # Start with a moderate step size
        sigma = 1.0
        
        # Adaptive (1+1)-ES
        # We'll use a simple adaptive strategy:
        # - If we improve, increase sigma slightly (explore more)
        # - If we don't improve, decrease sigma (exploit more)
        
        no_improve_count = 0
        max_no_improve = 30
        
        while not check_deadline():
            # Generate a single offspring
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            
            x_trial = x_best + sigma * direction
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            used_calls += 1
            
            if f_trial < f_best:
                # Improvement: accept and slightly increase sigma
                f_best = f_trial
                x_best = x_trial.copy()
                sigma *= 1.05  # Slight increase to explore more
                no_improve_count = 0
            else:
                # No improvement: decrease sigma
                sigma *= 0.95
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Stagnation: try a larger jump to escape local minimum
                    # Use a random restart with larger step
                    sigma = 0.5
                    no_improve_count = 0
                    
                    # Try a few random jumps
                    for _ in range(5):
                        if check_deadline():
                            break
                        x_jump = x_best + rng.standard_normal(dim) * sigma * 2
                        x_jump = np.clip(x_jump, -5, 5)
                        f_jump = func(x_jump)
                        used_calls += 1
                        
                        if f_jump < f_best:
                            f_best = f_jump
                            x_best = x_jump.copy()
                            sigma = 0.3
                            break
                    
                    # If still stuck, try even larger jumps
                    if no_improve_count > 0:
                        sigma = 1.0
        
        return x_best, f_best
