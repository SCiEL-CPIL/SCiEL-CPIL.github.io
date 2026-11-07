"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        
        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        
        # Multi-start configuration
        num_starts = 5
        # Allocate budget roughly equally, but leave some for the best one to polish
        # We will run starts sequentially, but if one stagnates, we move to the next.
        
        # Generate initial points
        # Use a few well-spread points and some random ones
        init_points = []
        # Center point
        init_points.append(np.zeros(self.dim))
        # Random points
        for _ in range(num_starts - 1):
            init_points.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle to avoid bias towards the first ones if they are bad?
        # Actually, evaluating all first gives us a baseline.
        # But we have a budget constraint.
        # Let's evaluate the first one, then decide.
        
        # Strategy:
        # 1. Evaluate all init_points (5 calls).
        # 2. Sort them.
        # 3. Run (1+1)-ES from the best.
        # 4. If it stagnates, run from the next best.
        
        if self.budget < num_starts:
            # Very small budget, just random search
            for i in range(self.budget):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x
            return global_best_x, global_best_f

        # Evaluate initial points
        candidates = []
        for x in init_points:
            if time.time() >= deadline:
                break
            fx = func(x)
            candidates.append((fx, x))
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Run (1+1)-ES from each candidate, starting with the best
        # We have used len(candidates) evaluations so far.
        # Remaining budget = self.budget - len(candidates)
        
        for fx, x in candidates:
            if self.budget - len(candidates) <= 0:
                break
            if time.time() >= deadline:
                break
            
            # Run local (1+1)-ES
            x_current = x.copy()
            f_current = fx
            sigma = 1.0
            
            # Limit the number of steps for this specific start to avoid wasting budget on a bad local min
            # Allow up to remaining_budget steps, but check for stagnation
            steps_for_this_start = 0
            max_steps_for_this_start = self.budget - (len(candidates) + steps_for_this_start)
            
            while steps_for_this_start < (self.budget - len(candidates)):
                if time.time() >= deadline:
                    break
                
                y = np.clip(x_current + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                steps_for_this_start += 1
                
                if fy < f_current:
                    x_current, f_current = y, fy
                    sigma *= 1.3
                    if f_current < global_best_f:
                        global_best_f = f_current
                        global_best_x = x_current.copy()
                else:
                    sigma *= 0.85
                    # Check for stagnation: if sigma is very small, we are stuck
                    if sigma < 1e-4:
                        break
                
                # If we have exhausted the budget for this specific allocation or total budget
                if len(candidates) + steps_for_this_start >= self.budget:
                    break
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
