"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate-wise optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Fixed set of step sizes for multi-scale search
        # Start large to jump between basins, end small for precise convergence
        step_sizes = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625]

        # Define deterministic starting points to ensure coverage of the box [-5, 5]^5
        # We use a stratified grid of starting points.
        # With dim=5, a full grid is expensive, so we pick a few representative centers.
        # Centers: 0, +/-2, +/-4
        # To keep it deterministic and simple, we'll cycle through a few key points.
        start_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.array([0.0, 2.0, -2.0, 4.0, -4.0]),
            np.array([2.0, 0.0, -4.0, 4.0, 0.0]),
            np.array([-2.0, 4.0, 0.0, -4.0, 2.0]),
        ]

        # For each starting point, perform a deterministic coordinate-wise hill climb
        # using the predefined step sizes.
        # We allocate the budget among these starts. 
        # 12 starts. 500 budget. ~40 calls per start if we just divide, but we can stop early.
        
        # Strategy: For each start, try all step sizes in order.
        # For each step size, do a coordinate-wise search.
        # If we find a better point, update x.
        # Stop a start if we run out of budget or time.
        
        for start_idx, start_x in enumerate(start_points):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            x = start_x.copy()
            
            # Initial evaluation
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Coordinate-wise search with shrinking steps
            for step in step_sizes:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                improved = True
                # We might need multiple passes for each step size to converge
                # But to save budget, let's do a single pass per step size first, 
                # or a few passes if we have budget.
                # Given 500 budget and 12 starts, we have ~40 calls per start.
                # 10 step sizes * 5 dims = 50 calls per full sweep.
                # So we can afford roughly 1 full sweep per start if we are careful, 
                # or fewer starts with more sweeps.
                
                # Let's do a "greedy" coordinate search:
                # For each dimension, try moving +step and -step.
                # If improvement, keep it.
                
                # We will do this for each step size.
                # To maximize progress, we stop the current start if no improvement 
                # was made in the entire sweep for a given step size? 
                # No, Lunacek can have flat regions or require small steps after large ones.
                # So we proceed through step sizes regardless, but we can break early 
                # if we hit the global minimum vicinity (e.g. f < -50).
                
                for _ in range(2): # 2 passes per step size to ensure convergence
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    for d in range(dim):
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Try positive step
                        trial_x = x.copy()
                        trial_x[d] += step
                        if -5 <= trial_x[d] <= 5:
                            fy = func(trial_x)
                            calls_made += 1
                            if fy < fx:
                                x = trial_x
                                fx = fy
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                        
                        # Try negative step
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break
                            
                        trial_x = x.copy()
                        trial_x[d] -= step
                        if -5 <= trial_x[d] <= 5:
                            fy = func(trial_x)
                            calls_made += 1
                            if fy < fx:
                                x = trial_x
                                fx = fy
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()

                # If we have found a very good solution, stop early
                if best_f < -80:
                    break
            
            # If we have found a very good solution, stop early
            if best_f < -80:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
