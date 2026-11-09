"""Rewritten optimizer for BBOB f20 (Schwefel) with robust (1+1)-ES and deadline guard."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with step-size adaptation, multiple restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        total_calls = 0

        # Use a fixed seed for reproducibility across instances
        rng = np.random.default_rng(42)

        # Strategy: multiple short runs with different initial points and sigmas,
        # then a final long run from the best point found.
        
        # Phase 1: Quick exploration with several restarts
        # We'll do up to 5 restarts, each with a fraction of the budget
        # but we need to be careful about total budget and time.
        
        # Let's do a single robust run with adaptive sigma and occasional restarts
        # if stuck, but keep it simple to avoid overhead.
        
        # Actually, for Schwefel, a good approach is a (1+1)-ES with 
        # careful sigma adaptation. Let's use a slightly more sophisticated 
        # sigma update: 1/5 success rule is standard, but we can also try
        # a simpler multiplicative update with bounds.
        
        # Let's try: start with a few random points, pick the best, then run ES from there.
        
        # Initial sampling: 10 random points
        n_init = 10
        init_points = rng.uniform(-5, 5, (n_init, self.dim))
        init_vals = []
        for i in range(n_init):
            x = init_points[i]
            fx = func(x)
            total_calls += 1
            init_vals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Now run (1+1)-ES from best_x
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Start with moderate sigma
        
        # Adapt sigma using 1/5 success rule
        # Keep a window of last 10 steps
        success_window = []
        
        remaining = self.budget - total_calls
        
        for i in range(remaining):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            total_calls += 1
            
            if fy < fx:
                # Accept
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                success_window.append(1)
                # Increase sigma
                sigma *= 1.1
            else:
                success_window.append(0)
                # Decrease sigma
                sigma *= 0.9
            
            # Keep window size at 10
            if len(success_window) > 10:
                success_window.pop(0)
            
            # Adapt sigma based on success rate
            if len(success_window) == 10:
                success_rate = sum(success_window) / 10.0
                if success_rate > 0.3:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
            
            # Bound sigma
            sigma = max(1e-8, min(sigma, 10.0))
        
        return best_x, best_f
