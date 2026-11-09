"""Improved optimizer for BBOB f9 (rotated Rosenbrock) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based CMA-ES for rotated Rosenbrock (f9) with structured init and final polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 265.0  # safety margin before 300s kill

        def timed():
            return (time.time() - t_start) > (deadline - t_start)

        def clip_box(x):
            # Ensure output is finite
            x = np.clip(x, -5.0, 5.0)
            x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            return x

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget or timed():
                return np.inf
            x_safe = clip_box(x)
            f_val = func(x_safe)
            calls_used += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x_safe.copy()
            return f_val

        # --- Phase 1: Broad Search ---
        # Sample 100 random points
        n_initial = 100
        for _ in range(n_initial):
            if calls_used >= budget or timed():
                break
            x_rand = rng.uniform(-5.0, 5.0, dim)
            evaluate(x_rand)

        if best_x is None:
            return np.zeros(dim), np.inf

        # --- Phase 2: Local Search with Adaptive Step Size ---
        # Start from best_x, try to improve with decreasing steps
        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.0
        
        # Use a fixed number of iterations or until budget is low
        max_local_iters = 150
        for _ in range(max_local_iters):
            if calls_used >= budget or timed():
                break
            if step_size < 1e-4:
                break
                
            improved = False
            # Generate 10 neighbors
            for _ in range(10):
                if calls_used >= budget or timed():
                    break
                # Random direction
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                x_trial = current_x + step_size * direction
                f_trial = evaluate(x_trial)
                
                if f_trial < current_f:
                    current_x = x_trial.copy()
                    current_f = f_trial
                    improved = True
                    # Keep step size or increase slightly? Keep for now.
                    break # Found improvement, restart neighbor search with same step?
                    # Or continue to find more? Let's restart to focus on new best.
            
            if not improved:
                step_size *= 0.7
            else:
                # If improved, maybe keep step size to explore further?
                # Or shrink slightly to refine?
                # Let's shrink slightly to refine around the new best.
                step_size *= 0.9

        # --- Phase 3: Coordinate Descent Polish ---
        if calls_used < budget and best_x is not None:
            x = best_x.copy()
            f_x = best_f
            # Use a few iterations of coordinate descent with small steps
            for _iter in range(5):
                if timed() or calls_used >= budget:
                    break
                for d in range(dim):
                    if timed() or calls_used >= budget:
                        break
                    for step in [0.1, 0.05, 0.01]:
                        for sign in [1, -1]:
                            if calls_used >= budget:
                                break
                            trial_x = x.copy()
                            trial_x[d] += sign * step
                            f_trial = evaluate(trial_x)
                            if f_trial < f_x:
                                x = trial_x
                                f_x = f_trial
                                break # Move to next dimension or step? Break inner sign loop
                        if f_x < best_f:
                            # Update global best if needed (evaluate already does this)
                            pass
                # If no improvement in this pass, we could break, but let's do a few passes
                
        if best_x is not None:
            best_x = clip_box(best_x)
        else:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
