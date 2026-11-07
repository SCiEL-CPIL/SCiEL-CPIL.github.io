import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        def eval_func(x):
            nonlocal calls_made, best_x, best_fx
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Random sampling to find good basins
        # We allocate a portion of the budget to initial exploration
        # For multimodal functions, landing in the right basin is key
        n_initial = min(20, total_budget // 5)
        
        initial_points = []
        for i in range(n_initial):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            initial_points.append((fx, x))
        
        # Sort initial points by fitness
        initial_points.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead from the best initial point
        # Nelder-Mead is efficient for unimodal functions (Sphere, Rosenbrock)
        # and can polish local optima for multimodal ones
        
        def nelder_mead(func_eval, x0, max_iter, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.1):
            dim = len(x0)
            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            simplex_fx = [func_eval(x0)]
            
            # Create other vertices
            for i in range(dim):
                new_x = x0.copy()
                if x0[i] != 0:
                    new_x[i] += 0.1 * abs(x0[i])
                else:
                    new_x[i] += 0.1
                new_x = np.clip(new_x, -5.0, 5.0)
                simplex.append(new_x)
                simplex_fx.append(func_eval(new_x))
            
            # Order simplex by fitness
            order = np.argsort(simplex_fx)
            simplex = [simplex[i] for i in order]
            simplex_fx = [simplex_fx[i] for i in order]
            
            for iteration in range(max_iter):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                fx_r = func_eval(x_r)
                
                if fx_r < simplex_fx[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    fx_e = func_eval(x_e)
                    
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        simplex_fx[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        simplex_fx[-1] = fx_r
                elif fx_r < simplex_fx[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_fx[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < simplex_fx[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        fx_c = func_eval(x_c)
                        
                        if fx_c <= fx_r:
                            simplex[-1] = x_c
                            simplex_fx[-1] = fx_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_fx[i] = func_eval(simplex[i])
                    else:
                        # Inside contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        fx_c = func_eval(x_c)
                        
                        if fx_c < simplex_fx[-1]:
                            simplex[-1] = x_c
                            simplex_fx[-1] = fx_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_fx[i] = func_eval(simplex[i])
                
                # Re-order simplex
                order = np.argsort(simplex_fx)
                simplex = [simplex[i] for i in order]
                simplex_fx = [simplex_fx[i] for i in order]
                
                # Check for convergence
                if abs(simplex_fx[0] - simplex_fx[-1]) < 1e-10:
                    break
            
            return simplex[0], simplex_fx[0]
        
        # Run Nelder-Mead from the best initial point
        if initial_points and len(initial_points) > 0:
            best_initial_fx, best_initial_x = initial_points[0]
            x_nm, fx_nm = nelder_mead(eval_func, best_initial_x, max_iter=100)
            
            # If we found a better solution, continue with more NM iterations
            if fx_nm < best_fx:
                best_fx = fx_nm
                best_x = x_nm.copy()
            
            # If budget remains, do more Nelder-Mead from the best point
            while calls_made < total_budget and time.time() < deadline:
                # Check if we're still improving
                x_nm, fx_nm = nelder_mead(eval_func, best_x, max_iter=20)
                if fx_nm >= best_fx:
                    break
                best_fx = fx_nm
                best_x = x_nm.copy()
        
        # Phase 3: If budget remains, try more random starts with NM polish
        remaining_budget = total_budget - calls_made
        if remaining_budget > 10:
            n_more_starts = min(5, remaining_budget // 30)
            for i in range(n_more_starts):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = eval_func(x)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                
                # NM polish
                x_nm, fx_nm = nelder_mead(eval_func, best_x, max_iter=20)
                if fx_nm < best_fx:
                    best_fx = fx_nm
                    best_x = x_nm.copy()
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_fx = eval_func(best_x)
        
        return best_x, best_fx
