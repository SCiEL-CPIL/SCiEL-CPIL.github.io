"""Hybrid Global Screening with Adaptive Coordinate Descent for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using a hybrid strategy:
    1. Global Screening: Evaluate diverse starting points.
    2. Local Refinement: Adaptive Coordinate Descent (CD-AS) with line search.
    3. Restarts: If local search stagnates, restart from a new random point.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 285.0  # Safety margin for 300s limit

        n_calls = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Screening
        # We spend ~20% of the budget on diverse starts to identify good basins.
        n_starts = max(10, int(budget * 0.2))
        
        # Deterministic + Random starts
        starts = []
        starts.append(np.zeros(dim))
        for scale in [1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        # Evaluate starts
        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            safe_func(s)

        if best_x is None:
            # Fallback if something went wrong
            return np.zeros(dim), np.inf

        # Phase 2: Adaptive Coordinate Descent with Restarts
        # We will run local CD-AS from the current best point.
        # If it stagnates, we pick a new random point and try again.
        
        max_restarts = 10
        restart_count = 0
        current_x = best_x.copy()
        current_f = best_f
        
        # Parameters for CD-AS
        initial_step = 1.0
        min_step = 1e-8
        stagnation_threshold = 20  # No improvement in 20 full cycles (20 * dim evaluations)
        cycle_count = 0
        
        while n_calls[0] < budget and time.time() < deadline:
            if restart_count > max_restarts:
                break
                
            # Local Search Loop
            step_size = initial_step
            no_improve_count = 0
            local_best_x = current_x.copy()
            local_best_f = current_f
            
            # If this is a restart, we might want a slightly larger initial step
            if restart_count > 0:
                step_size = rng.uniform(0.5, 2.0)
            
            while n_calls[0] < budget and time.time() < deadline:
                improved_in_cycle = False
                
                for i in range(dim):
                    if n_calls[0] >= budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative steps along dimension i
                    # We use a simple 3-point check or a shrinking step
                    # To save budget, we try one step in the best direction found previously, 
                    # but here we just try +/- step_size
                    
                    # Check positive step
                    x_plus = local_best_x.copy()
                    x_plus[i] += step_size
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = safe_func(x_plus)
                    
                    if f_plus < local_best_f:
                        local_best_f = f_plus
                        local_best_x = x_plus
                        improved_in_cycle = True
                        no_improve_count = 0
                        # If improved, we might shrink step slightly to refine
                        step_size *= 0.8 
                    else:
                        # Check negative step
                        x_minus = local_best_x.copy()
                        x_minus[i] -= step_size
                        x_minus = np.clip(x_minus, -5.0, 5.0)
                        f_minus = safe_func(x_minus)
                        
                        if f_minus < local_best_f:
                            local_best_f = f_minus
                            local_best_x = x_minus
                            improved_in_cycle = True
                            no_improve_count = 0
                            step_size *= 0.8
                        else:
                            # No improvement in this direction
                            # If step size is very small, we are stuck
                            if step_size > min_step:
                                # Try a smaller step in the next iteration? 
                                # For now, we just proceed to next dimension.
                                # We shrink step size globally if no improvement in whole cycle.
                                pass

                # End of dimension cycle
                if not improved_in_cycle:
                    no_improve_count += 1
                    # Shrink step size if no improvement
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                else:
                    no_improve_count = 0
                
                cycle_count += 1
                if no_improve_count > stagnation_threshold:
                    break

            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            # Decide on restart
            # If we broke out of the inner loop due to stagnation or min step
            # and we still have budget, restart from a new random point
            # but keep the global best.
            
            if n_calls[0] < budget and time.time() < deadline:
                restart_count += 1
                # Pick a new random starting point for the next local search
                # We can bias it towards the current best or purely random
                # Pure random is better for global exploration in multi-modal functions
                current_x = rng.uniform(-4.5, 4.5, dim)
                # Evaluate the new start
                current_f = safe_func(current_x)
                # If the new start is better than the local best we just found, 
                # it becomes the new local best for the next iteration
                if current_f < local_best_f:
                    local_best_x = current_x.copy()
                    local_best_f = current_f

        return best_x, best_f
# EVOLVE-BLOCK-END
