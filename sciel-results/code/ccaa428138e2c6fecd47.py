"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = np.random.uniform(-5, 5, dim)
        best_fx = func(best_x)
        n_calls += 1
        
        def time_left():
            return deadline - time.time()
            
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Coarse Grid Search to identify the basin
        # The step ellipsoidal function has a global minimum that is separated from
        # local minima by plateaus. A coarse grid helps locate the correct region.
        grid_size = 3
        grid_points = np.meshgrid(*[np.linspace(-5, 5, grid_size) for _ in range(dim)])
        grid_points = np.array(grid_points).reshape(-1, dim)
        
        for x in grid_points:
            if n_calls >= budget or time_left() < 10:
                break
            fx = func(x)
            n_calls += 1
            update_best(x, fx)

        # Phase 2: Coordinate Descent with Brent's Method
        # Exploit the separable-like structure in the local neighborhood.
        # Iterate through dimensions, performing 1D minimization along each axis.
        # This is efficient for ellipsoidal functions once in the basin.
        
        # Determine how many passes we can afford
        # Each pass costs roughly dim * (cost of 1D minimization)
        # Let's reserve some budget for final polish.
        
        max_passes = 20
        
        for p in range(max_passes):
            if n_calls >= budget or time_left() < 5:
                break
                
            improved = False
            for d in range(dim):
                if n_calls >= budget or time_left() < 5:
                    break
                    
                # Define 1D function for axis d
                def f_1d(t):
                    nonlocal n_calls
                    if n_calls >= budget:
                        return best_fx
                    x_temp = best_x.copy()
                    x_temp[d] = t
                    fx_temp = func(x_temp)
                    n_calls += 1
                    if fx_temp < best_fx:
                        best_fx = fx_temp
                        best_x = x_temp.copy()
                        return fx_temp
                    return fx_temp
                
                # Use Brent's method
                # Bounds: clip to box, but allow some expansion if at boundary?
                # For now, stick to box bounds.
                a = -5.0
                b = 5.0
                
                # If we are at the boundary, restrict the search range to avoid wasting calls outside
                # But since func clips, it's safe. However, Brent might evaluate outside.
                # Let's restrict to [max(-5, best_x[d]-2), min(5, best_x[d]+2)] to be efficient
                # and assume the minimum is close.
                
                # Actually, let's just use the full box but be careful.
                # If best_x[d] is close to boundary, restrict.
                if best_x[d] < -4:
                    b = -3
                elif best_x[d] > 4:
                    a = 3
                    
                try:
                    res = minimize_scalar(f_1d, bounds=(a, b), method='brent', options={'maxiter': 20})
                    if res.success:
                        # res.fun is the value, res.x is the coordinate
                        # We already updated best_x and best_fx inside f_1d if it improved
                        # But f_1d might have been called multiple times.
                        # We need to ensure best_x reflects the best found.
                        # The update_best logic inside f_1d handles this.
                        pass
                except Exception:
                    pass
                    
                # Check if we improved
                # Since f_1d updates best_x and best_fx, we just continue.
                # To detect stagnation, we could track if best_fx changed.
                # But for now, just run the passes.

        # Phase 3: Final Polish with Nelder-Mead
        remaining = budget - n_calls
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', 
                               options={'maxfev': remaining, 'maxiter': remaining, 
                                        'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
