import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(f_val):
                return best_f
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Coarse grid search to identify multiple basins
        # 3^5 = 243 points, spaced 2.5 apart in [-5, 5]
        # This gives good coverage of the box for Rosenbrock-like functions
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([-5 + i*2.5, -5 + j*2.5, -5 + k*2.5, -5 + l*2.5, -5 + m*2.5])
                            grid_points.append(x)

        # Shuffle grid points with a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = [grid_points[i] for i in perm]

        # Evaluate grid points
        basin_centers = []
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            f_val = eval_func(x)
            if np.isfinite(f_val):
                basin_centers.append((f_val, x))

        if not basin_centers:
            return best_x, best_f

        # Sort by fitness and take top basins
        basin_centers.sort(key=lambda t: t[0])
        
        # Determine how many basins to refine based on remaining budget
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # Allocate remaining budget among top basins
        # Use Nelder-Mead style refinement for each basin
        # For dim=5, Nelder-Mead needs about dim*(dim+1)/2 = 15 initial points
        # plus iterations. Let's use a simple local search.

        # Select top K basins
        # Each basin refinement will use roughly remaining/K calls
        # With Nelder-Mead, we need ~15 + 5*20 = 115 calls per basin for good convergence
        # So with 500-243=257 remaining, we can do about 2 basins well
        # Let's be more conservative and do 1-2 basins

        n_basins = min(len(basin_centers), max(1, remaining // 80))
        
        # Refine each selected basin using a simple gradient-free local search
        # We'll use a pattern search / Nelder-Mead hybrid

        for bi in range(n_basins):
            if calls_used >= budget or time.time() > deadline:
                break
            
            f0, x0 = basin_centers[bi]
            
            # Nelder-Mead simplex method
            # Initial simplex: x0 and x0 + delta*e_i for i=1..dim
            delta = 0.5  # Initial step size
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate initial simplex
            simplex_f = []
            for sx in simplex:
                sx = np.clip(sx, -5, 5)
                fx = eval_func(sx)
                simplex_f.append(fx)
            
            # Nelder-Mead iterations
            max_iter = 50
            for it in range(max_iter):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = eval_func(xc)
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = eval_func(simplex[i])
                
                # Check convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

        return best_x, best_f
