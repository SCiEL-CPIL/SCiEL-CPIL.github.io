import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initial state
        # Start at origin, which is often close to the hidden shift in BBOB f1
        # If not, the search will expand.
        x = np.zeros(dim)
        f_x = func(x)
        calls_made = 1
        
        # Track best
        best_x = x.copy()
        best_f = f_x
        
        # Pattern Search / Coordinate Descent with Adaptive Step Size
        # Strategy:
        # 1. Try to improve in each coordinate direction.
        # 2. If improvement is found, expand step size (cautiously) to accelerate.
        # 3. If no improvement in a full cycle, shrink step size to refine.
        # 4. Stop if step size is tiny or budget/time exhausted.
        
        step_size = 4.0  # Initial step size, covers most of the box [-5, 5]
        min_step = 1e-10
        expand_factor = 1.5
        shrink_factor = 0.5
        
        # Directions: +/- for each coordinate
        # We will iterate through coordinates 0..dim-1, trying +step and -step
        
        # Optimization: Pre-allocate direction vectors to avoid repeated allocation
        # But since we only move one coordinate at a time, we can just modify x in place.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            improvement_found = False
            
            for i in range(dim):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try positive direction
                x_new = x.copy()
                x_new[i] += step_size
                # Clip to box [-5, 5]
                x_new = np.clip(x_new, -5.0, 5.0)
                
                # If the step didn't actually move us (e.g., already at bound), skip evaluation?
                # Or evaluate to be safe? If x_new[i] == x[i], f is same.
                if not np.allclose(x_new, x):
                    f_new = func(x_new)
                    calls_made += 1
                    
                    if f_new < f_x:
                        x = x_new
                        f_x = f_new
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        improvement_found = True
                        # Expand step size for next iteration? 
                        # For coordinate descent, expanding immediately might cause oscillation.
                        # Let's expand after a full cycle of success.
                        # Or, we can just keep the step size and see if we can go further.
                        # Actually, standard pattern search expands if a step succeeds.
                        # Let's expand step_size for the *next* coordinate? 
                        # No, let's keep it simple: if we improve, we stay. 
                        # We will expand the step size only if we improve in ALL coordinates in a cycle? 
                        # That's too strict. 
                        # Let's expand step_size by a small factor if we improve, but cap it.
                        # Actually, for Sphere, moving towards the optimum reduces the gradient magnitude.
                        # A fixed step might be fine, but adaptive is better.
                        # Let's try: if improved, step_size *= 1.1 (slight expansion)
                        # But if we hit the bound, we shouldn't expand.
                        if x_new[i] < 5.0 and x_new[i] > -5.0:
                            step_size = min(step_size * 1.1, 10.0) # Cap expansion
                    else:
                        # No improvement in + direction, try - direction
                        x_new2 = x.copy()
                        x_new2[i] -= step_size
                        x_new2 = np.clip(x_new2, -5.0, 5.0)
                        
                        if not np.allclose(x_new2, x):
                            f_new2 = func(x_new2)
                            calls_made += 1
                            
                            if f_new2 < f_x:
                                x = x_new2
                                f_x = f_new2
                                if f_x < best_f:
                                    best_f = f_x
                                    best_x = x.copy()
                                improvement_found = True
                                if x_new2[i] < 5.0 and x_new2[i] > -5.0:
                                    step_size = min(step_size * 1.1, 10.0)
                            else:
                                # No improvement in either direction for this coordinate
                                pass
                        else:
                            # At bound, no move possible in this direction
                            pass
                else:
                    # At bound, skip
                    pass
            
            # End of coordinate cycle
            if not improvement_found:
                # Shrink step size
                step_size *= shrink_factor
                if step_size < min_step:
                    break
            else:
                # If we found improvement, we might have expanded step_size.
                # To prevent oscillation or large steps that overshoot, 
                # we could reset step_size to a moderate value if it got too big?
                # For Sphere, larger steps are good if moving towards center.
                # But if we are close, large steps might bounce.
                # Let's add a check: if step_size is very large and we are improving, 
                # it's fine. If we start failing, we shrink.
                pass
                
        return best_x, best_f
