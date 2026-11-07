"""Rewritten optimizer for BBOB f14 (Different powers) using multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin: 330s limit
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling - use more diverse starting points
        # Sample from a grid + random points to cover the search space
        init_points = []
        # Origin
        init_points.append(np.zeros(d))
        # Random points
        for _ in range(15):
            init_points.append(rng.uniform(-5, 5, d))
        
        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = try_eval(best_x)
            
        if calls_used >= max_calls or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead
        # Run Nelder-Mead from several promising starting points
        # Sort initial points by fitness and use top ones as NM starts
        
        # Collect all evaluated points with their fitness
        evaluated = []
        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            # We already evaluated these, but let's track them
            # Actually, we need to re-evaluate or store them. Let's just use the best few.
        
        # Use the best initial point as the primary NM start
        # And run a few additional NM starts from other good points
        
        nm_starts = [best_x.copy()]
        # Add a few other diverse points as additional NM starts
        # We'll pick points that are far from the best
        for p in init_points:
            if len(nm_starts) >= 5:
                break
            p_clipped = np.clip(p, -5, 5)
            if np.linalg.norm(p_clipped - best_x) > 1.0:
                nm_starts.append(p_clipped.copy())
        
        # If we don't have enough diverse starts, add random ones
        while len(nm_starts) < 5:
            nm_starts.append(rng.uniform(-5, 5, d))
        
        # Run Nelder-Mead from each start
        for start in nm_starts:
            if calls_used >= max_calls or time.time() > deadline - 3:
                break
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Initialize simplex
            simplex = [start.copy()]
            fvals = [best_f]
            
            # Create initial simplex points
            for i in range(d):
                new_point = start.copy()
                # Step size: 0.05 if start is non-zero, 0.05 in direction i
                step = 0.05 if abs(start[i]) < 0.01 else 0.05 * np.sign(start[i])
                new_point[i] += step
                new_point = np.clip(new_point, -5, 5)
                simplex.append(new_point)
                fx = try_eval(new_point)
                if fx is None:
                    break
                fvals.append(fx)
            
            if calls_used >= max_calls or len(fvals) < d + 1:
                continue
            
            # Main Nelder-Mead loop
            max_nm_iter = 200  # Limit iterations per NM run
            for iter in range(max_nm_iter):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5, 5)
                
                # Worst point
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = try_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_s = try_eval(simplex[i])
                            if f_s is None:
                                break
                            fvals[i] = f_s
                        if calls_used >= max_calls:
                            break
                
                # Check for convergence
                if len(fvals) >= 2:
                    f_range = max(fvals) - min(fvals)
                    if f_range < 1e-10:
                        break
            
            # Update global best
            if fvals and min(fvals) < best_f:
                best_f = min(fvals)
                # Find the best point in the final simplex
                order = np.argsort(fvals)
                best_x = simplex[order[0]].copy()

        # Phase 3: Final polish with (1+1)-ES
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            remaining = max_calls - calls_used
            for i in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                if sigma < 1e-10:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
