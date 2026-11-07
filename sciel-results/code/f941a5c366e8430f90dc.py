"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start Coordinate Descent with adaptive step sizes and stagnation-triggered random jumps.
    
    Strategy:
    1. Initialize multiple candidate points (multi-start).
    2. For each candidate, perform Coordinate Descent:
       - Systematically sweep each dimension to find the local minimum along that axis.
       - Use adaptive step sizes (binary search-like refinement) to converge quickly.
       - If no improvement is found after a few sweeps, trigger a "Random Jump" to escape local minima.
    3. Track the global best across all starts and the entire budget.
    
    This approach is highly effective for axis-aligned functions (Sphere, Rastrigin) and uses
    jumps to handle rotated/multimodal functions (Rosenbrock, Gallagher).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Multi-start configuration
        # We don't want to spend all budget on one start. 
        # Let's allocate budget dynamically.
        # Start with a few initial points spread out.
        # For dim=5, let's try 5-10 initial starts.
        
        # Initial points:
        # 1. Origin (0,0,0,0,0)
        # 2. Random points in [-5, 5]^5
        # 3. Maybe some structured points?
        
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Add random points
        n_random_starts = 8
        for _ in range(n_random_starts):
            initial_points.append(rng.uniform(-5, 5, dim))
            
        # Shuffle to avoid bias towards origin if it's bad, but origin is good for Sphere.
        # Actually, for Sphere, origin is perfect. For others, it's just a point.
        # Let's keep origin first, then randoms.
        
        # We will process these starts sequentially.
        # Each start gets a portion of the budget.
        # If a start makes progress, we give it more budget.
        # If it stagnates, we move to the next start.
        
        # Let's define a function for one "CD session" starting from a point.
        # It returns the best found and calls used.
        
        def coordinate_descent_session(start_x, max_calls):
            nonlocal calls_made, best_x, best_fx
            
            x = start_x.copy()
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            if calls_made >= budget or time.time() > deadline:
                return x, fx, calls_made
            
            # Adaptive step sizes for each dimension
            # Initial step size: 2.0 (covers half the range)
            steps = np.ones(dim) * 2.0
            
            # Coordinate Descent Loop
            # We sweep dimensions 0..dim-1 repeatedly.
            # In each sweep, for each dimension, we try to minimize along that axis.
            
            no_improve_sweeps = 0
            max_sweeps = 50 # Cap on sweeps per start to prevent infinite loop if stuck
            
            for sweep in range(max_sweeps):
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                improved_in_sweep = False
                
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try to minimize along dimension d
                    # Use a local search along the axis:
                    # Try x[d] +/- step[d]
                    # If improvement, move there and shrink step.
                    # If no improvement, shrink step and try again? Or just shrink and continue.
                    
                    current_val = x[d]
                    current_fx = fx
                    best_local_x = x.copy()
                    best_local_fx = fx
                    
                    # Binary search-like refinement
                    # We want to find the minimum in the neighborhood.
                    # Simple approach: try positive and negative steps.
                    
                    # Try positive
                    x_trial = x.copy()
                    x_trial[d] = np.clip(current_val + steps[d], -5, 5)
                    if not np.allclose(x_trial, x): # Avoid redundant eval if step too small
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < best_local_fx:
                            best_local_fx = f_trial
                            best_local_x = x_trial.copy()
                            improved_in_sweep = True
                            # Shrink step for next time in this dim
                            steps[d] *= 0.5
                    else:
                        # Step too small, shrink significantly
                        steps[d] *= 0.5
                    
                    # Try negative
                    x_trial = x.copy()
                    x_trial[d] = np.clip(current_val - steps[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < best_local_fx:
                            best_local_fx = f_trial
                            best_local_x = x_trial.copy()
                            improved_in_sweep = True
                            steps[d] *= 0.5
                    else:
                        steps[d] *= 0.5
                    
                    # If we found a better point, move there
                    if best_local_fx < current_fx:
                        x = best_local_x
                        fx = best_local_fx
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        # No improvement in this dim with current step.
                        # Shrink step to explore finer scale?
                        # Or keep step and move to next dim?
                        # Shrinking is safer for convergence, but might miss wider minima.
                        # Let's shrink slightly if we've been stuck.
                        if steps[d] > 1e-4:
                            steps[d] *= 0.7
                        else:
                            # Step is very small, this dim is likely optimized locally.
                            pass

                if not improved_in_sweep:
                    no_improve_sweeps += 1
                else:
                    no_improve_sweeps = 0
                    
                # Stagnation Trigger: If no improvement for a few sweeps, do a Random Jump
                if no_improve_sweeps >= 3:
                    # Random Jump
                    # Jump size: proportional to current step size or fixed?
                    # Let's try a jump of size 1.0 to 2.0 in random direction.
                    # Or jump in a random dimension by a larger amount.
                    
                    # Strategy: Pick a random dimension and jump by a random amount in [-2, 2]
                    # Or perturb all dims slightly.
                    # For multimodal, a larger jump is better.
                    
                    jump_dim = rng.integers(0, dim)
                    jump_size = rng.uniform(0.5, 2.0)
                    sign = rng.choice([-1, 1])
                    
                    x_jump = x.copy()
                    x_jump[jump_dim] += sign * jump_size
                    x_jump = np.clip(x_jump, -5, 5)
                    
                    f_jump = func(x_jump)
                    calls_made += 1
                    
                    if f_jump < fx:
                        x = x_jump
                        fx = f_jump
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        # Reset steps? Or keep them?
                        # Keep them, but maybe increase the jumped dim's step?
                        steps[jump_dim] = 1.0
                        no_improve_sweeps = 0
                    else:
                        # Jump failed, shrink steps further to try finer local search?
                        # Or try another jump?
                        # Let's just continue, steps will shrink naturally.
                        pass
                
                # If steps are all very small, we've converged locally.
                if np.all(steps < 1e-4):
                    break

            return x, fx, calls_made

        # Execute Multi-Start
        # We will iterate through initial_points.
        # If budget is low, we might not finish all starts.
        
        for i, start_point in enumerate(initial_points):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Allocate budget for this start
            # Simple heuristic: remaining budget / remaining starts
            # But if we find a good point, we want to exploit it more.
            # Let's just let the CD loop run until it stagnates or budget is out.
            # The CD loop has internal max_sweeps and step shrinking, so it will terminate.
            
            # To prevent one start from eating all budget if it's slow to stagnate,
            # we can cap the calls per start.
            # But the CD loop is efficient.
            
            coordinate_descent_session(start_point, budget - calls_made)
            
            # Check time/budget
            if time.time() > deadline:
                break

        # Final Polish: If we have budget left, do a final fine-tuning CD on the best point found.
        if calls_made < budget and best_x is not None:
            # Reset steps to be very small for polishing
            # Or just run CD again with smaller initial steps?
            # The previous CD already shrinks steps. 
            # Let's just run one more session with the best_x, but force smaller steps?
            # The function above re-initializes steps to 2.0. 
            # We can modify it or just call it again. It will quickly shrink.
            
            # Actually, the previous session might have stopped early due to max_sweeps or stagnation.
            # Let's do a final "polish" pass.
            
            def polish_session():
                nonlocal calls_made, best_x, best_fx
                
                if best_x is None:
                    return
                
                x = best_x.copy()
                fx = best_fx
                steps = np.ones(dim) * 0.1 # Start with smaller steps for polish
                
                # Similar loop but focused on refinement
                for sweep in range(20):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    improved = False
                    for d in range(dim):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        
                        # Try +/- step
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x[d] + steps[d], -5, 5)
                        if not np.allclose(x_trial, x):
                            f_trial = func(x_trial)
                            calls_made += 1
                            if f_trial < fx:
                                fx = f_trial
                                x = x_trial.copy()
                                improved = True
                                steps[d] *= 0.5
                        else:
                            steps[d] *= 0.5
                        
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x[d] - steps[d], -5, 5)
                        if not np.allclose(x_trial, x):
                            f_trial = func(x_trial)
                            calls_made += 1
                            if f_trial < fx:
                                fx = f_trial
                                x = x_trial.copy()
                                improved = True
                                steps[d] *= 0.5
                        else:
                            steps[d] *= 0.5
                        
                    if not improved:
                        break # Converged
                    
                    if np.all(steps < 1e-5):
                        break
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            
            polish_session()

        # If no calls were made (shouldn't happen), return origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
