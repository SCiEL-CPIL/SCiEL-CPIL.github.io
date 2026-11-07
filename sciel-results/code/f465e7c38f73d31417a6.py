"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic multi-scale coordinate descent for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 2.0

        # Initial guess: 0 is a safe center
        x = np.zeros(dim)
        f = func(x)
        calls += 1
        update_best(x, f)

        # Multi-scale coordinate descent (Pattern Search)
        # Start with a large stride to quickly locate the basin of attraction
        stride = 4.0
        
        # We iterate scales until stride becomes very small or budget/time runs out
        while calls < budget and time_left() and stride > 1e-5:
            
            # At each scale, we try to improve the current point by moving
            # along each coordinate axis by +/- stride.
            # If we find an improvement, we accept it and keep the stride (or shrink slightly if we want to be aggressive).
            # If we fail to find any improvement in a full pass, we shrink the stride.
            
            improved = False
            for coord in range(dim):
                if calls >= budget or not time_left():
                    break
                
                # Try + direction
                trial_x = x.copy()
                trial_x[coord] = np.clip(x[coord] + stride, -5, 5)
                trial_f = func(trial_x)
                calls += 1
                
                if trial_f < f:
                    x, f = trial_x, trial_f
                    update_best(x, f)
                    improved = True
                    # We can continue trying other directions or restart the loop?
                    # Restarting the coordinate loop is often better for pattern search to catch coupled moves.
                    # But for efficiency, let's just continue to the next coordinate in this pass.
                    # Actually, a standard pattern search restarts the axis loop if a move is accepted.
                    # Let's restart to ensure we don't miss a better direction in a previous axis.
                    break 
                
                # Try - direction
                if calls >= budget or not time_left():
                    break
                trial_x = x.copy()
                trial_x[coord] = np.clip(x[coord] - stride, -5, 5)
                trial_f = func(trial_x)
                calls += 1
                
                if trial_f < f:
                    x, f = trial_x, trial_f
                    update_best(x, f)
                    improved = True
                    break # Restart axis loop

            if not improved:
                # No improvement found with current stride, shrink it
                stride *= 0.5

        # Final polish: Use a very small stride to squeeze out the last bit of precision
        # This acts as a local refinement
        if calls < budget and time_left():
            final_stride = 1e-4
            # Simple local search with tiny steps
            for _ in range(5): # A few iterations of tiny steps
                if calls >= budget or not time_left():
                    break
                improved = False
                for coord in range(dim):
                    if calls >= budget or not time_left():
                        break
                    trial_x = x.copy()
                    trial_x[coord] = np.clip(x[coord] + final_stride, -5, 5)
                    trial_f = func(trial_x)
                    calls += 1
                    if trial_f < f:
                        x, f = trial_x, trial_f
                        update_best(x, f)
                        improved = True
                        break
                    
                    if calls >= budget or not time_left():
                        break
                    trial_x = x.copy()
                    trial_x[coord] = np.clip(x[coord] - final_stride, -5, 5)
                    trial_f = func(trial_x)
                    calls += 1
                    if trial_f < f:
                        x, f = trial_x, trial_f
                        update_best(x, f)
                        improved = True
                        break
                if not improved:
                    final_stride *= 0.5
                    if final_stride < 1e-8:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
