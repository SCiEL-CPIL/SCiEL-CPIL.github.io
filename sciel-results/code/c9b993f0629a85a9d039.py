import numpy as np
import time

class Optimizer:
    """
    Grid-based Multi-Start Nelder-Mead Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Phase 1: Deterministic sparse grid search (3^5 = 243 points) to identify top basins.
    - Phase 2: Independent Nelder-Mead local optimizations from the top 5 grid points.
    - Robust to rotation and non-separability by exploring multiple basins simultaneously.
    - Strict time and call budget management.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin for 300s limit
        
        n = self.dim
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        # Phase 1: Sparse Grid Search
        # Use 3 levels per dimension: -3, 0, 3
        # This covers a large portion of [-5, 5]^5 and is deterministic.
        levels = [-3.0, 0.0, 3.0]
        grid_points = np.array(np.meshgrid(*([levels]*n))).T.reshape(-1, n)
        
        # Evaluate all grid points
        grid_results = []
        for x in grid_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            f_x = func(x)
            calls_made += 1
            grid_results.append((f_x, x.copy()))
            
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
        
        if not grid_results:
            return best_x, best_f
        
        # Sort by fitness and take top 5
        grid_results.sort(key=lambda item: item[0])
        top_points = [x for f, x in grid_results[:5]]
        
        # Phase 2: Nelder-Mead Local Optimization
        # Allocate remaining budget to Nelder-Mead
        remaining_budget = self.budget - calls_made
        # Nelder-Mead typically needs O(dim * max_iterations) calls. 
        # With dim=5 and ~250 calls left, we can do ~50 iterations per point.
        # But we have 5 points, so ~10 iterations each? Or adaptively allocate.
        
        # Let's implement a simple Nelder-Mead that respects the global budget.
        # We will run Nelder-Mead for each top point, sharing the global call count.
        
        for start_x in top_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Initialize simplex
            simplex = [start_x.copy()]
            f_simplex = [func(start_x)]
            calls_made += 1
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = start_x.copy()
            
            for i in range(1, n+1):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                x_i = start_x.copy()
                # Perturb dimension i
                if x_i[i] < 4.5:
                    x_i[i] += 0.5
                else:
                    x_i[i] -= 0.5
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls_made += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(simplex) < n + 1:
                continue
                
            # Nelder-Mead Iterations
            max_iter = 100
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Sort simplex
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Replace worst
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + 2 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            f_shrink = func(simplex[i])
                            calls_made += 1
                            f_simplex[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
        
        return best_x, best_f
