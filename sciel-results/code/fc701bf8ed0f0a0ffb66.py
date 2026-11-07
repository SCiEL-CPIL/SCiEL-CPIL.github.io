"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Cooperative Co-evolutionary (CC) Random-Direction Search for Rastrigin-rotated f15.
    
    Strategy:
    1. Decompose the 5D problem into 5 1D sub-problems.
    2. Each sub-problem uses a random direction search with discrete, log-spaced step sizes.
    3. This avoids the continuous step-size adaptation pitfalls of (1+1)-ES (overshooting/undershooting).
    4. The discrete steps allow precise exploitation of the periodic structure of Rastrigin.
    5. A global restart mechanism handles multimodality by resetting the search if no improvement is seen for a while.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin for 300s limit
        
        # Use a fixed seed for reproducibility in the sub-searches if needed, 
        # but random is fine since the evaluator runs instances separately.
        rng = np.random.default_rng(42)

        # Discrete step sizes: from large (exploration) to small (exploitation)
        # Rastrigin period is 2pi. We want steps that can jump across periods 
        # and steps that can refine within a period.
        # Log-spaced steps covering ~2 orders of magnitude.
        step_sizes = np.logspace(np.log10(5.0), np.log10(1e-4), 50)
        
        # Current best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Initial evaluation
        best_f = func(best_x)
        calls_made += 1

        # Cooperative Co-evolution loop
        # We iterate through dimensions. For each dimension, we try to improve 
        # the current best by moving along that dimension only.
        
        no_global_improve = 0
        max_global_stall = 50 # If no global improvement in 50 dimension-updates, restart
        
        # To handle rotation, we don't just use coordinate axes. 
        # We use a set of random orthonormal vectors as "coordinate systems".
        # However, for simplicity and speed, we first try standard coordinates.
        # If that stagnates, we try random rotations.
        
        # Generate a few random rotation matrices for diverse search directions
        rotations = [np.eye(dim)]
        for _ in range(4):
            # Generate random orthogonal matrix
            Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
            rotations.append(Q)

        # Main loop
        while calls_made < budget and time.time() < deadline:
            improved = False
            
            # Try each rotation frame
            for rot in rotations:
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # For each dimension in the rotated frame
                for d in range(dim):
                    if calls_made >= budget or time.time() < deadline:
                        break
                    
                    # Search along the d-th column of the rotation matrix
                    direction = rot[:, d]
                    
                    # Random direction sign
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    direction = sign * direction
                    
                    # Try discrete steps from large to small
                    # Start from a medium step, not the largest, to avoid wasting budget
                    # unless we are in a "restart" mode.
                    
                    # Adaptive starting index: if we haven't improved in a while, start bigger
                    if no_global_improve > 20:
                        start_idx = max(0, len(step_sizes) - 30) # Start with smaller steps if stuck? 
                        # Actually, if stuck, we might be in a local min. 
                        # Let's stick to a sweep: try large to small.
                        start_idx = 0
                    else:
                        start_idx = 10 # Start with medium steps
                    
                    for i in range(start_idx, len(step_sizes)):
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        
                        step = step_sizes[i]
                        candidate = best_x + step * direction
                        candidate = np.clip(candidate, -5.0, 5.0)
                        
                        f_cand = func(candidate)
                        calls_made += 1
                        
                        if f_cand < best_f:
                            best_f = f_cand
                            best_x = candidate.copy()
                            improved = True
                            no_global_improve = 0
                            # If we improved, we might want to continue refining 
                            # in this same direction with smaller steps
                            # But the loop naturally goes to smaller steps.
                        else:
                            # If we tried a large step and failed, smaller steps might work.
                            # If we tried a very small step and failed, we are likely at a local min
                            # for this direction. We can break early if step is very small.
                            if step < 1e-3:
                                break
            
            # Update stall counter
            if not improved:
                no_global_improve += 1
                # If we have stalled significantly, we might want to jump to a new random point
                # to escape the local basin.
                if no_global_improve > max_global_stall:
                    # Random restart
                    new_x = rng.uniform(-4, 4, dim)
                    f_new = func(new_x)
                    calls_made += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_x.copy()
                        no_global_improve = 0
                    else:
                        # Even if not better, we reset the stall counter to allow more search
                        # in the current best's neighborhood before restarting again
                        no_global_improve = 0
            else:
                no_global_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
