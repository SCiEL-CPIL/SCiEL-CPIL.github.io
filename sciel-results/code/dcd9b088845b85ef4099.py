"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate-descent hybrid with adaptive step sizes for BBOB-style Schwefel."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Start from center of the box
        x = np.zeros(dim)
        try:
            f = func(x)
        except Exception:
            f = np.inf
        
        evals = 1
        best_x = x.copy()
        best_f = f
        
        # Adaptive step sizes per dimension
        step_sizes = np.ones(dim) * 2.0
        
        # Coordinate descent with line search along each axis
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Try each coordinate direction
            for d in range(dim):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try positive and negative directions
                for direction in [1, -1]:
                    if evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Line search: try a few step sizes
                    for step_factor in [1.0, 0.5, 0.25, 0.1]:
                        if evals >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        new_x = x.copy()
                        new_x[d] += direction * step_sizes[d] * step_factor
                        np.clip(new_x, -5, 5, out=new_x)
                        
                        try:
                            new_f = func(new_x)
                        except Exception:
                            new_f = np.inf
                        evals += 1
                        
                        if new_f < f:
                            x = new_x
                            f = new_f
                            step_sizes[d] *= 1.5  # Increase step if improvement
                            
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            break  # Stop line search for this direction
                        else:
                            # No improvement, decrease step size
                            step_sizes[d] *= 0.7
                            if step_sizes[d] < 0.01:
                                break
                # End direction loop
            # End coordinate loop
            
            # Add small random perturbation to escape local minima
            if evals < budget and time.time() < deadline:
                # Every 50 evaluations, add a random perturbation
                if evals % 50 < dim:
                    perturbation = rng.normal(0, 0.5, size=dim)
                    new_x = x + perturbation
                    np.clip(new_x, -5, 5, out=new_x)
                    
                    try:
                        new_f = func(new_x)
                    except Exception:
                        new_f = np.inf
                    evals += 1
                    
                    if new_f < f:
                        x = new_x
                        f = new_f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    # If perturbation didn't help, keep current x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
