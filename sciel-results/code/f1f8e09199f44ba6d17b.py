"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Grid-Search + Adaptive Coordinate Descent optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # 300s limit. 280s margin is safe.
        deadline = start_time + 280.0
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline or calls >= budget:
                return best_f
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Exploration via Coarse Grid
        # Weierstrass is multimodal. A grid helps find good basins.
        # Dim=5. A 3^5 = 243 point grid is too many (budget 500).
        # A 2^5 = 32 point grid is cheap.
        # Let's use a deterministic set of points covering [-4, 4]^5.
        
        # Generate grid points: -4, 0, 4 in each dimension? 
        # 3^5 = 243 calls. This uses ~50% budget.
        # Maybe 2^5 = 32 calls is better to save budget for refinement.
        # Let's use a mix: center + corners + midpoints.
        # Actually, for Weierstrass, the global minimum is often near the center or shifted.
        # Let's use a deterministic "random" set of 20 points + center.
        
        rng = np.random.default_rng(42)
        
        # 1. Center
        points = [np.zeros(dim)]
        
        # 2. Deterministic random points in [-4, 4]
        n_init = 20
        for _ in range(n_init):
            p = rng.uniform(-4, 4, dim)
            points.append(p)
            
        # Evaluate initialization points
        for p in points:
            if time.time() > deadline or calls >= budget:
                break
            wrapped_func(p)
            
        # Phase 2: Adaptive Coordinate Descent (CD-AS)
        # Cycle through dimensions, performing a line search along each.
        # Weierstrass is smooth, so local structure is good.
        # We will run this for the remaining budget.
        
        # Initial step size
        step_sizes = np.full(dim, 1.0)
        
        # We will iterate until budget or time is exhausted.
        # To ensure progress, we do full sweeps of dimensions.
        
        # Optimization: Instead of generic line search which might be slow,
        # use a fixed number of evaluations per dimension per sweep.
        # E.g., for each dim, try x +/- step.
        # If improvement, shrink step. If not, shrink step or change sign?
        # Standard CD: 
        # for d in 0..dim-1:
        #   try x[d] + step[d]
        #   try x[d] - step[d]
        #   keep best
        #   if improved: step[d] *= 0.7 (shrink)
        #   else: step[d] *= 1.5 (expand) or reset?
        
        # Let's implement a robust version:
        # Each "sweep" costs dim * 2 evaluations (plus current point check? no, we just evaluate neighbors).
        # Actually, we evaluate x_new. We don't need to re-evaluate x_old if we track it.
        
        # We have ~470 calls left.
        # dim=5. 2 evals/dim/sweep => 10 evals/sweep.
        # ~47 sweeps possible.
        
        # To handle the "stagnation" or "getting stuck in local minima" of CD:
        # If no improvement in a full sweep, we can try a larger random jump (restart-like)
        # or just let it converge. Given the budget, convergence is likely.
        
        # Let's refine the CD logic:
        # We track current_x and current_f (which is best_f/best_x if we started CD from best).
        # We should start CD from the best point found in Phase 1.
        
        curr_x = best_x.copy()
        curr_f = best_f
        
        # We will run CD until budget/time runs out.
        # We need to be careful: if CD fails to improve, we might waste budget.
        # But CD is very cheap per call.
        
        # Adaptive step size logic:
        # If we improve, shrink step.
        # If we don't improve after trying +/-, shrink step significantly to try to find a smaller improvement.
        # If step becomes too small, we might be stuck.
        
        # Let's use a simple strategy:
        # For each dimension:
        #   x_try = curr_x.copy()
        #   x_try[d] += step_sizes[d]
        #   f_try = wrapped_func(x_try)
        #   if f_try < curr_f:
        #       curr_x = x_try; curr_f = f_try
        #       step_sizes[d] *= 0.8
        #   else:
        #       x_try = curr_x.copy()
        #       x_try[d] -= step_sizes[d]
        #       f_try = wrapped_func(x_try)
        #       if f_try < curr_f:
        #           curr_x = x_try; curr_f = f_try
        #           step_sizes[d] *= 0.8
        #       else:
        #           # No improvement in this dim
        #           step_sizes[d] *= 0.5 # Shrink to look for smaller steps
        #           if step_sizes[d] < 1e-8:
        #               # Reset step size for this dim to a medium value to escape?
        #               # Or just stop trying this dim for a while?
        #               step_sizes[d] = 0.1
        #           # We didn't move, so curr_x is unchanged.
        
        # This logic uses 2 calls per dimension per sweep.
        # If we improve, we used 1 call.
        
        # Let's optimize the loop to check time/budget frequently.
        
        sweep_count = 0
        while time.time() < deadline and calls < budget:
            improved_in_sweep = False
            for d in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                    
                # Try positive step
                x_try = curr_x.copy()
                x_try[d] += step_sizes[d]
                f_try = wrapped_func(x_try)
                
                if f_try < curr_f:
                    curr_x = x_try
                    curr_f = f_try
                    step_sizes[d] *= 0.8
                    improved_in_sweep = True
                else:
                    # Try negative step
                    if time.time() > deadline or calls >= budget:
                        break
                    x_try = curr_x.copy()
                    x_try[d] -= step_sizes[d]
                    f_try = wrapped_func(x_try)
                    if f_try < curr_f:
                        curr_x = x_try
                        curr_f = f_try
                        step_sizes[d] *= 0.8
                        improved_in_sweep = True
                    else:
                        # No improvement
                        step_sizes[d] *= 0.5
                        if step_sizes[d] < 1e-9:
                            step_sizes[d] = 0.5 # Reset to medium to allow escape if stuck in a flat region?
                            # Actually, for Weierstrass, small steps are good for precision.
                            # Let's keep it small.
                            step_sizes[d] = 1e-6

            sweep_count += 1
            
            # If no improvement in a full sweep, we might be at a local minimum or step is too small.
            # If step is too small, we are converged.
            # If step is not small, we are stuck.
            # We can add a "kick" if we stagnate for a few sweeps.
            # But for now, simple CD is the structural change requested.
            
            # To prevent infinite loops of tiny steps, we can break if all steps are very small.
            if np.all(step_sizes < 1e-7):
                # We are converged to high precision.
                # We can use remaining budget to do a final random restart check?
                # Or just break.
                # Given AOCC rewards early progress, and we are converged, breaking is fine.
                # However, if we have lots of budget left, maybe we should check if a random jump helps?
                # Let's add a simple restart check if we have > 50 calls left.
                if calls < budget - 50 and time.time() < deadline - 10:
                     # Try a few random points to see if we found a better global min
                     for _ in range(10):
                         if calls >= budget: break
                         p = rng.uniform(-4, 4, dim)
                         f_p = wrapped_func(p)
                         if f_p < curr_f:
                             curr_x = p
                             curr_f = f_p
                             # Reset steps for new start
                             step_sizes = np.full(dim, 1.0)
                             break
                break

        # Update global best
        if curr_f < best_f:
            best_f = curr_f
            best_x = curr_x

        return best_x, best_f
# EVOLVE-BLOCK-END
