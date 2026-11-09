"""Rosenbrock Valley Tracker with Adaptive Coordinate Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Rosenbrock (f8).
    Strategy: 
    1. Coarse global search to locate the basin (handling rotation/shift).
    2. Custom 'Valley Tracker' local search. This exploits the structure of Rosenbrock 
       where the minimum lies in a narrow valley. It uses adaptive step sizes that 
       expand rapidly when moving along the valley (low curvature) and contract 
       when moving across it (high curvature). This is significantly more efficient 
       than generic methods like Nelder-Mead for this specific function class.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        def time_left():
            return deadline - time.time()

        # State variables
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return float('inf')
            calls_made += 1
            return func(x)

        # --- Phase 1: Global Exploration ---
        # The minimizer is in [-4, 4]^5. 
        # We perform a coarse grid search to find the best starting point.
        # A 3^5 = 243 point grid is affordable and provides good coverage.
        # We also add the origin and a few random points to handle potential edge cases.
        
        # Generate grid points
        # Using linspace from -4.5 to 4.5 with 3 points: -4.5, 0, 4.5
        # This might be too coarse if the valley is shifted, but it's a start.
        # Let's use a slightly denser grid on the likely region or random sampling.
        # Given 500 budget, 243 grid points leaves ~250 for local search.
        
        # To be safer against rotation, we mix grid and random.
        # Let's do 100 random points + 100 grid points.
        rng = np.random.default_rng(42)
        
        # Random sampling
        for _ in range(100):
            if time_left() < 20: break
            x = rng.uniform(-4.5, 4.5, dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Coarse Grid Sampling (3^5 = 243 points is too many if we already did 100 randoms? 
        # Total 343. Leaves 157 for local. That's tight but okay for a fast local search.)
        # Let's stick to a smaller grid: 2 points per dim? No, that's just corners.
        # Let's do a 3x3x3x3x3 grid but skip if time is tight.
        
        # Actually, 100 randoms + 100 more targeted randoms might be better than a fixed grid 
        # if the rotation is arbitrary. But the valley is long. 
        # Let's do a "Halton" or quasi-random sequence for better space-filling.
        
        # Implement simple Sobol-like or just more randoms for the rest of the exploration phase.
        # Let's reserve 200 calls for exploration (100 done, 100 to go).
        for _ in range(100):
            if time_left() < 20: break
            x = rng.uniform(-4.5, 4.5, dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # --- Phase 2: Local Refinement (Valley Tracker) ---
        # We have ~300 calls left.
        # Algorithm:
        # 1. Initialize step sizes.
        # 2. Iterate over dimensions.
        # 3. For each dim, try to move in the direction that reduces f.
        # 4. If f decreases, expand step size in that direction (valley tracking).
        # 5. If f increases, shrink step size and try opposite direction or smaller step.
        # 6. This mimics the behavior of following the curved valley.

        x = best_x.copy()
        f = best_f
        steps = np.ones(dim) * 0.5
        min_step = 1e-8
        max_step = 5.0
        contraction = 0.5
        expansion = 1.5

        # We will run cycles until time or budget runs out.
        # A cycle is one pass through all dimensions.
        # Each dimension attempt costs 1-2 calls.
        
        no_improvement_count = 0
        max_no_improvement = 50 # If we stagnate, we might need a small random jump to escape local minima (though Rosenbrock has few)
        
        while time_left() > 5 and calls_made < budget:
            improved = False
            for i in range(dim):
                if time_left() < 5 or calls_made >= budget:
                    break
                
                # Try positive step
                x_new = x.copy()
                x_new[i] += steps[i]
                f_new = safe_call(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    improved = True
                    # Expand step in this direction (valley tracking)
                    steps[i] = min(steps[i] * expansion, max_step)
                else:
                    # Try negative step
                    x_new = x.copy()
                    x_new[i] -= steps[i]
                    f_new = safe_call(x_new)
                    
                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True
                        # Expand step in this direction
                        steps[i] = min(steps[i] * expansion, max_step)
                    else:
                        # Neither direction improved significantly with current step
                        # Shrink step size
                        steps[i] = max(steps[i] * contraction, min_step)
                        # If step is very small, we are likely at a minimum for this coordinate
                        # Try a tiny random perturbation in this coordinate to escape flat regions?
                        # Or just accept it.
                        
            if improved:
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    # Stagnation. Do a small random restart around current best.
                    # This helps if we got stuck in a local minimum (unlikely for Rosenbrock, but possible due to rotation/shift mapping issues)
                    # Or if steps became too small.
                    # Let's reset steps to a moderate value and do a few random steps.
                    steps = np.ones(dim) * 0.2
                    for _ in range(10):
                        if time_left() < 5 or calls_made >= budget: break
                        x_trial = best_x + rng.normal(0, 0.1, dim)
                        f_trial = safe_call(x_trial)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                            x = best_x.copy()
                            f = best_f
                    no_improvement_count = 0
                    # Reset x to best found
                    x = best_x.copy()
                    f = best_f

        return best_x, best_f

# EVOLVE-BLOCK-END
