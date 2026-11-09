"""Multi-Start Deterministic Coordinate Descent for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Deterministic Coordinate Descent optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        calls_used = 0
        
        def evaluate(x):
            nonlocal calls_used, best_f_global, best_x_global
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_used += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx

        def coordinate_descent(x0, f0):
            """
            Perform coordinate descent starting from x0.
            Returns best_x, best_f found during this phase.
            """
            nonlocal calls_used, best_f_global, best_x_global
            
            x = x0.copy()
            f = f0
            best_local_x = x.copy()
            best_local_f = f
            
            # Adaptive step size
            step = 2.0
            no_improve_count = 0
            
            # Coordinate descent loop
            while calls_used < budget:
                if time.time() > deadline:
                    break
                
                improved = False
                for i in range(d):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step
                    f_plus = evaluate(x_plus)
                    
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try negative step
                        x_minus = x.copy()
                        x_minus[i] -= step
                        f_minus = evaluate(x_minus)
                        
                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            improved = True
                            no_improve_count = 0
                        else:
                            # No improvement in this coordinate, maybe shrink step?
                            # Or just move to next coordinate.
                            # For step functions, if we are in a flat region, 
                            # we might need to jump. But CD usually handles this 
                            # by eventually finding a drop.
                            # If we are stuck, we might need to reduce step size 
                            # to find the edge of the step, or increase to jump over it.
                            # Given f7 is step ellipsoidal, the function is flat 
                            # within each "step". The minimum is at the center of 
                            # the lowest step. 
                            # Actually, f7 is defined such that the function value 
                            # depends on the maximum of |x_i - shift_i| after rotation.
                            # It's a "step" function. 
                            # If we are in a flat region, moving won't change f. 
                            # We need to find the boundary.
                            # A simple CD might get stuck in a flat region if 
                            # the gradient is 0. 
                            # However, f7 is not smooth. 
                            # Let's add a small random perturbation if stuck for a while.
                            pass
                    
                    # If we improved, we stay. If not, we check if we should shrink step.
                    # Actually, standard CD updates step if no improvement over a full cycle.
                    
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > d:
                        # Stuck. Try shrinking step to find the edge of the current step
                        step *= 0.5
                        if step < 1e-4:
                            break
                        # Reset no_improve_count to allow a few more tries with smaller step
                        no_improve_count = 0
                    else:
                        # Try increasing step to jump over the current flat region?
                        # For step functions, increasing step might help escape local flat areas.
                        # But if we are at the minimum, we don't want to jump away.
                        # Let's stick to shrinking for now to refine.
                        pass
                else:
                    # If we improved, we might want to keep the step size or increase slightly?
                    # Usually keep it.
                    pass
                
                # Update global best
                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x.copy()
                
                # Check for convergence
                if step < 1e-5:
                    break

            return best_local_x, best_local_f

        # Phase 1: Global Search with Multi-Start
        # We have 500 calls. 
        # Let's use ~100 calls for global exploration (10 restarts * 10 points) 
        # and the rest for local refinement.
        
        # Generate initial candidates
        # 1. Center
        # 2. Random points in [-4, 4]
        
        num_restarts = 10
        calls_per_restart_exploration = 5 # Just pick a point
        
        # Actually, let's just pick 10 random points, evaluate them, and pick the best 3 for refinement.
        # Or just run CD from the best of the first few.
        
        candidates = []
        fvals = []
        
        # Add center
        c0 = np.zeros(d)
        f0 = evaluate(c0)
        candidates.append(c0)
        fvals.append(f0)
        
        # Add random points
        for _ in range(9):
            c = rng.uniform(-4.0, 4.0, d)
            f = evaluate(c)
            candidates.append(c)
            fvals.append(f)
            
        # Sort by function value
        indices = np.argsort(fvals)
        
        # Pick top 3 for local refinement
        # If we have budget left, we can refine the best one deeply.
        # Let's refine the best one first.
        
        best_idx = indices[0]
        x_start = candidates[best_idx].copy()
        f_start = fvals[best_idx]
        
        # Run Coordinate Descent from the best start
        if calls_used < budget:
            coordinate_descent(x_start, f_start)
            
        # If we still have budget, try the second best
        if calls_used < budget and len(indices) > 1:
            second_idx = indices[1]
            # Avoid re-evaluating if it's the same point (unlikely)
            x_start2 = candidates[second_idx].copy()
            f_start2 = fvals[second_idx]
            # Only run if it's significantly better or if we haven't converged
            # Just run it, CD will stop if no improvement
            coordinate_descent(x_start2, f_start2)
            
        # If we still have budget, try the third best
        if calls_used < budget and len(indices) > 2:
            third_idx = indices[2]
            x_start3 = candidates[third_idx].copy()
            f_start3 = fvals[third_idx]
            coordinate_descent(x_start3, f_start3)
            
        # If we still have budget, do a final fine-tuning on the global best
        if calls_used < budget and best_x_global is not None:
            # Use a very small step size for final polishing
            # We can just call coordinate_descent again with a small initial step
            # But our CD function has a fixed initial step of 2.0.
            # Let's modify CD to accept an initial step or just rely on the shrinking.
            # The shrinking in CD will handle it.
            # However, if we are already at the minimum, CD will shrink step and stop.
            # This is efficient.
            pass

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
