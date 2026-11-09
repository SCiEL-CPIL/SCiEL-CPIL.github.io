"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES with random initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0
        points_evaluated = []  # List of (x, f)

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            points_evaluated.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ===== Phase 1: Global Exploration =====
        # Sample 20 points to identify promising basins
        num_initial = 20
        initial_points = []
        
        # Always include the origin
        initial_points.append(np.zeros(n))
        
        # Sample from Gaussian distribution centered at 0
        # Std 2.0 covers most of the [-5, 5] box
        while len(initial_points) < num_initial:
            p = rng.normal(0, 2.0, n)
            p = np.clip(p, -5, 5)
            initial_points.append(p)
            
        for p in initial_points:
            if not check_time() or eval_count >= budget:
                break
            fp = func(p)
            try_update(p, fp)
            
        # If we ran out of budget, return best
        if eval_count >= budget or not check_time():
            return best_x, best_f

        # Sort evaluated points by function value (ascending)
        points_evaluated.sort(key=lambda item: item[1])
        
        # ===== Phase 2: Local Exploitation =====
        # Take the top 5 best points as starting points for (1+1)-ES
        top_k = 5
        # Ensure we don't pick more starts than we have points
        top_k = min(top_k, len(points_evaluated))
        
        starts = [p[0] for p in points_evaluated[:top_k]]
        
        # Calculate budget for each ES run
        remaining_budget = budget - eval_count
        budget_per_es = max(10, remaining_budget // top_k)
        
        for start_x in starts:
            if not check_time() or eval_count >= budget:
                break
                
            # Get the function value of the start point
            # It is already in points_evaluated
            # Find it efficiently (or just assume it's the one we picked)
            # Since we sorted, the first top_k are the best.
            # We can just look up the value.
            fx = None
            for px, pf in points_evaluated:
                if np.allclose(px, start_x):
                    fx = pf
                    break
            if fx is None:
                # Fallback (should not happen)
                fx = func(start_x)
                try_update(start_x, fx)

            x = start_x.copy()
            
            # (1+1)-ES Parameters
            sigma = 0.5
            no_improve = 0
            max_no_improve = 15
            steps = 0
            max_steps = budget_per_es
            
            while steps < max_steps and eval_count < budget and check_time():
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                
                fy = func(y)
                try_update(y, fy)
                steps += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase sigma on success
                    sigma = min(sigma * 1.2, 5.0)
                    no_improve = 0
                else:
                    # Decrease sigma on failure
                    sigma *= 0.8
                    no_improve += 1
                    
                # Stopping criteria
                if sigma < 1e-5 or no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
