import numpy as np
import time

class Optimizer:
    """
    Derivative-free Nelder-Mead optimizer with a sparse global lattice search.
    
    Strategy:
    1. Use a sparse global lattice to identify promising basins of attraction early.
    2. Run Nelder-Mead from the best points found in the lattice.
    3. Nelder-Mead adapts the simplex shape to navigate the narrow Rosenbrock valley
       without the overhead and noise of finite-difference gradients.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                # Return a large value to signal termination without making an invalid call
                return 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter=100, tol=1e-6):
            nonlocal calls_made, best_f, best_x
            
            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            f_vals = [func(x0)]
            
            for i in range(dim):
                x_new = x0.copy()
                # Initial step size: 0.05 for coordinates that are 0, 0.05 + 0.05*|x0[i]| otherwise
                if abs(x0[i]) < 1e-8:
                    x_new[i] += 0.05
                else:
                    x_new[i] += 0.05 * abs(x0[i])
                simplex.append(x_new)
                f_vals.append(func(x_new))
            
            # Sort simplex
            simplex = np.array(simplex)
            f_vals = np.array(f_vals)
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Check convergence
                if np.max(f_vals) - np.min(f_vals) < tol:
                    break
                
                # Compute centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = func(x_r)
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Reflection is good
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = func(x_c)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = func(x_c)
                    
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                
                # Re-sort
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]
                
                # Update global best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
            
            return best_x, best_f

        # Step 1: Sparse global lattice search
        # Use a 2x2x2x2x2 lattice (32 points) to cover the box [-5, 5]^5
        # This is cheap (32 calls) and helps identify broad basins.
        lattice_points = []
        for i in range(2 ** dim):
            point = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    point[j] = 5.0
                else:
                    point[j] = -5.0
            lattice_points.append(point)
        
        # Also add the center and some random points
        lattice_points.append(np.zeros(dim))
        rng = np.random.default_rng(42)
        for _ in range(10):
            lattice_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all lattice points
        lattice_results = []
        for x in lattice_points:
            if calls_made >= budget or time.time() > deadline - 5:
                break
            f = tracked_func(x)
            lattice_results.append((f, x))
        
        # Sort by function value
        lattice_results.sort(key=lambda t: t[0])
        
        # Step 2: Run Nelder-Mead from the best few lattice points
        # Use the top 3 points as starting points for Nelder-Mead
        num_restarts = 3
        for i in range(num_restarts):
            if calls_made >= budget or time.time() > deadline - 5:
                break
            
            if i < len(lattice_results):
                x0 = lattice_results[i][1]
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            nelder_mead(tracked_func, x0, max_iter=50, tol=1e-8)
        
        # Step 3: If budget remains, run Nelder-Mead from the current best point
        if calls_made < budget and best_x is not None and time.time() < deadline - 5:
            nelder_mead(tracked_func, best_x, max_iter=50, tol=1e-10)
        
        # Final fallback
        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = tracked_func(best_x)
        
        return best_x, best_f
