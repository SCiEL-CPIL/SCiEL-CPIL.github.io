import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        
        # Helper to check time and budget
        def check_ok():
            return calls_made < total_budget and time.time() < deadline - 1.0

        # Helper to evaluate
        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_f
            if not check_ok():
                return np.inf
            f_val = func(x)
            calls_made += 1
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x.copy()
            return f_val

        # Phase 1: Deterministic Multi-Scale Grid Search
        # Goal: Cover the space reasonably to find a good basin quickly.
        # For dim=5, a full grid of 3^5=243 points is too many for "early" if we want to save budget for NM.
        # But 1.5499 -> 2.785 improvement came from grid+NM.
        # Let's use a coarse grid first, then refine the best region?
        # Or just a fixed set of points that covers the box well.
        
        # Let's use a 3-level grid:
        # Level 1: 3 points per dim -> 3^5 = 243 points. This is almost half the budget (500).
        # Maybe too many? 
        # If we spend 243 calls on grid, we have 257 left for NM.
        # NM in 5D needs at least dim+1=6 points per simplex. 
        # With 257 calls, we can do roughly 257/6 ~ 40 steps per start.
        # If we start from top 5 grid points, that's 200 steps total. Feasible.
        
        # However, if the minimum is very sharp, we need more NM steps.
        # Let's try a slightly sparser grid or adaptive grid.
        # Actually, the insight said "43 points" for grid. 43 is not a power.
        # Maybe a Halton-like deterministic set was used? Or a specific pattern?
        # 3^3 * 5? No.
        # Let's stick to a principled grid. 
        # If we use 2 points per dim: 2^5 = 32 points. Very cheap.
        # If we use 3 points per dim: 243 points. Expensive.
        
        # Let's try a hybrid: 
        # 1. Evaluate center (0,0,0,0,0).
        # 2. Evaluate corners (2^5 = 32 points).
        # 3. Evaluate midpoints of edges? 
        # Total so far: 1 + 32 = 33.
        # This gives a very coarse view.
        
        # Better: Use a deterministic low-discrepancy sequence (like Halton) for the first ~100 points?
        # The previous program used Halton for 40% (200 points). 
        # The successful one used "43 points" grid.
        # Let's generate a grid of 3 points per dimension but prune it?
        # Or just use the 3^5 grid but stop early if time is tight?
        
        # Let's try the 3^5 grid (243 points) but only if we have enough budget.
        # If budget is 500, 243 is 48%. 
        # Let's see if we can do better with fewer grid points.
        # A 2^5 grid (32 points) is very fast. Then use the remaining 468 calls for NM.
        # With 468 calls, we can run NM for a very long time.
        # NM converges fast. 
        # Let's try: 
        # 1. 2^5 grid (32 points) + center (1) = 33 points.
        # 2. Sort by f.
        # 3. Run NM from top 10 points.
        
        # Wait, the insight said "43 points". 
        # 3^3 = 27. 27 + 16? 
        # Maybe it was a specific set.
        # Let's just use a 3-point grid per dimension but limited to the best sub-box?
        # No, let's keep it simple and robust.
        
        # Strategy:
        # 1. Coarse Grid: 3 points per dim (243 points). 
        #    If this is too many, we will run out of budget for NM.
        #    Let's check: 500 - 243 = 257.
        #    NM steps: 257 / (dim+1) ~ 257/6 = 42 steps per start.
        #    If we start from 5 best points: 5 * 42 = 210 steps. 
        #    This seems reasonable.
        
        # However, if the function is very noisy or has many local minima, we might need more starts.
        # Let's try a slightly smaller grid: 2 points per dim (32 points).
        # 500 - 32 = 468.
        # 468 / 6 = 78 steps per start.
        # We can start from top 10 points: 10 * 78 = 780 steps? No, budget is shared.
        # We just run NM until budget is exhausted or convergence.
        
        # Let's go with the 3^5 grid (243 points) to ensure we don't miss a peak far from the center.
        # But 243 is a lot. 
        # Let's try 3 points in first 3 dims, 2 in last 2? 3*3*3*2*2 = 108 points.
        # 500 - 108 = 392.
        # 392 / 6 = 65 steps per start.
        # Start from top 5: 5 * 65 = 325 steps.
        # This seems like a good balance.
        
        # Let's implement a generic grid generator.
        # dims_split = [3, 3, 3, 2, 2]
        
        grid_sizes = [3, 3, 3, 2, 2]
        if dim < 5:
            grid_sizes = grid_sizes[:dim]
            while len(grid_sizes) < dim:
                grid_sizes.append(2)
        
        points = []
        for i1 in range(grid_sizes[0]):
            for i2 in range(grid_sizes[1]):
                for i3 in range(grid_sizes[2]):
                    for i4 in range(grid_sizes[3]):
                        for i5 in range(grid_sizes[4]):
                            x = np.zeros(dim)
                            # Map indices to [-5, 5]
                            # If size is 3: -5, 0, 5
                            # If size is 2: -5, 5 (or -2.5, 2.5?)
                            # Let's use linspace.
                            
                            # We will generate all points and then evaluate them.
                            pass
        
        # Easier: use itertools.product
        import itertools
        ranges = [np.linspace(-5, 5, s) for s in grid_sizes]
        grid_points = list(itertools.product(*ranges))
        
        # Shuffle deterministically? No, keep order for reproducibility, but evaluate all.
        # To ensure "early progress", we should evaluate points that are likely good first?
        # Or just evaluate all grid points first.
        # The AOCC metric rewards early progress. 
        # If we evaluate 108 grid points, the first 108 calls are "exploration".
        # If we find the global min in call 50, that's great.
        
        # Let's evaluate the grid points.
        # To improve early progress, we can interleave grid evaluation with a few random/Halton points?
        # No, let's stick to the grid.
        
        grid_values = []
        for pt in grid_points:
            if not check_ok():
                break
            x = np.array(pt)
            f_val = evaluate(x)
            grid_values.append((f_val, x))
            
        if not grid_values:
            # Fallback if grid failed
            return global_best_x, global_best_f

        # Sort grid points by value
        grid_values.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Nelder-Mead
        # Start from the top K grid points.
        K = 5
        starts = [x for f, x in grid_values[:K]]
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter=100, tol=1e-8):
            nonlocal calls_made, global_best_x, global_best_f
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            calls_made += 1
            if f_simplex[0] < global_best_f:
                global_best_f = f_simplex[0]
                global_best_x = x0.copy()
                
            # Initialize other vertices
            step = 0.1
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += step
                simplex.append(x_new)
                f_simplex.append(func(x_new))
                calls_made += 1
                if f_simplex[-1] < global_best_f:
                    global_best_f = f_simplex[-1]
                    global_best_x = x_new.copy()
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if not check_ok():
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if abs(f_simplex[0] - f_simplex[-1]) < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < global_best_f:
                    global_best_f = f_r
                    global_best_x = x_r.copy()
                
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Replace worst
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < global_best_f:
                        global_best_f = f_e
                        global_best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls_made += 1
                        if f_c < global_best_f:
                            global_best_f = f_c
                            global_best_x = x_c.copy()
                        
                        if f_c < f_r:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_simplex[i] = func(simplex[i])
                                calls_made += 1
                                if f_simplex[i] < global_best_f:
                                    global_best_f = f_simplex[i]
                                    global_best_x = simplex[i].copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            if f_simplex[i] < global_best_f:
                                global_best_f = f_simplex[i]
                                global_best_x = simplex[i].copy()
            
            return global_best_x, global_best_f

        for start_x in starts:
            if not check_ok():
                break
            nelder_mead(func, start_x, max_iter=50)
            
        return global_best_x, global_best_f
