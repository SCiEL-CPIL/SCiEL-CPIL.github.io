"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Coordinate Descent (DFCD) with adaptive steps
    tailored for the bi-modal Lunacek bi-Rastrigin function f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_x, best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse global sampling to find promising regions
        # Use a small set of diverse starting points
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:dim]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0][:dim]),
        ]
        
        # Add a few random points for diversity
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, size=dim))

        # Evaluate all starts
        start_results = []
        for start_x in starts:
            if calls_made >= budget - 20:
                break
            fx = eval_point(start_x)
            start_results.append((start_x, fx))

        # Sort by function value, best first
        start_results.sort(key=lambda x: x[1])

        # Phase 2: Coordinate descent from the best starting points
        # Use adaptive step sizes that shrink based on progress
        max_starts_to_refine = min(3, len(start_results))
        
        for start_idx in range(max_starts_to_refine):
            if calls_made >= budget - 15:
                break
            
            x = start_results[start_idx][0].copy()
            fx = start_results[start_idx][1]
            
            # Adaptive coordinate descent
            step = 2.0
            no_improve_count = 0
            max_no_improve = 3
            
            while step >= 0.0005 and calls_made < budget - 10:
                improved = False
                
                for i in range(dim):
                    if calls_made >= budget - 10:
                        break
                    
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            no_improve_count = 0
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            no_improve_count = 0
                
                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    step *= 0.5
                    if no_improve_count >= max_no_improve:
                        break

        # Phase 3: Fine local refinement around the best point found
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_fx
            
            # Very fine steps for final polish
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            for step in fine_steps:
                if calls_made >= budget - 5:
                    break
                
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 5:
                        break
                    
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            
                if not improved:
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
