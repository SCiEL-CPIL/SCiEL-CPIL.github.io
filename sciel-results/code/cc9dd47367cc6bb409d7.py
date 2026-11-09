"""Portfolio search for BBOB f7 (Step ellipsoidal) with plateau-aware (1+1)-ES and random restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES with plateau-aware adaptation and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        # Initial broad exploration with a few random points
        n_initial = min(10, budget)
        for i in range(n_initial):
            if calls_made >= budget or time_left() < 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Main search: multiple (1+1)-ES restarts with plateau-aware sigma adaptation
        # For step ellipsoidal, the function is flat in some directions and steep in others.
        # We use a (1+1)-ES with success-based sigma adaptation and occasional restarts.
        
        max_restarts = 15
        calls_per_restart = max(30, budget // max_restarts)
        
        for restart in range(max_restarts):
            if calls_made >= budget or time_left() < 10.0:
                break
            
            # Choose starting point: mix of best-so-far perturbation and fresh random
            if restart == 0 and best_x is not None:
                x = best_x.copy()
            elif restart % 3 == 0 and best_x is not None:
                # Perturb best point
                x = best_x + 1.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
            else:
                # Fresh random start
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            plateau_count = 0
            max_plateau = 30  # restart if stuck for this many steps
            
            for step in range(calls_per_restart):
                if calls_made >= budget or time_left() < 5.0:
                    break
                
                # Generate trial point
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Improvement
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.25
                    plateau_count = 0
                elif fy == fx:
                    # Plateau - no change (common in step functions)
                    plateau_count += 1
                    # Small random nudge to escape plateau
                    x = x + 0.1 * sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    sigma *= 0.95
                else:
                    # Worse
                    sigma *= 0.85
                    plateau_count += 1
                
                # Adapt sigma based on plateau
                if plateau_count > max_plateau:
                    # Restart from current position with larger exploration
                    sigma = max(sigma, 2.0)
                    plateau_count = 0
                    # Try a larger step
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.3
                        plateau_count = 0
                    else:
                        sigma *= 0.8
                        plateau_count = 0
                
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.01, 10.0)

        # Final polish: fine-grained local search around best point
        if best_x is not None and calls_made < budget and time_left() > 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_steps = min(50, budget - calls_made)
            for _ in range(polish_steps):
                if calls_made >= budget or time_left() < 2.0:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
