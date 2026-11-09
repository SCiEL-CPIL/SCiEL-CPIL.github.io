"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Split budget: 20% for random sampling to find good basins, 80% for local refinement
        # Given the hidden shift/rotation, starting points need to be diverse.
        # Schaffers F7 is multi-modal, so local search from a single point is risky.
        
        n_starts = 10
        budget_per_start = max(10, (self.budget - n_starts) // n_starts)
        
        best_x = None
        best_f = float('inf')
        
        # Initial random sampling to identify promising regions
        # We use a portion of the budget to sample and pick the best few for NM
        sample_budget = min(50, self.budget // 10)
        samples = []
        for _ in range(sample_budget):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            samples.append((fx, x))
            
        # Sort samples and pick the best distinct ones (simple deduplication by distance)
        samples.sort()
        
        # Select starting points for Nelder-Mead
        start_points = []
        for fx, x in samples:
            if best_f > fx:
                best_f = fx
                best_x = x.copy()
                
            # Check if this point is sufficiently different from existing starts
            is_distinct = True
            for sx in start_points:
                if np.linalg.norm(x - sx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                start_points.append(x)
            if len(start_points) >= n_starts:
                break
                
        # If we didn't find enough distinct points, just use the best ones found
        if len(start_points) < n_starts:
            for fx, x in samples:
                if x not in start_points:
                    start_points.append(x)
                if len(start_points) >= n_starts:
                    break

        # Run Nelder-Mead from each start point
        for i, x0 in enumerate(start_points):
            if time.time() > deadline:
                break
                
            # Estimate remaining budget for this run
            # Roughly: (total_budget - used_so_far) / remaining_starts
            # But we have a fixed budget_per_start logic. 
            # Let's just run NM with a max_iter that fits in the time/budget.
            
            # Nelder-Mead implementation
            n = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            f_values = [func(x0)]
            
            # Initialize other vertices
            step_size = 0.5
            for j in range(n):
                x_j = x0.copy()
                if j == 0:
                    x_j[0] += step_size
                else:
                    x_j[j] += step_size
                simplex.append(x_j)
                f_values.append(func(x_j))
                
            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            max_iter = 200 # Cap iterations to avoid infinite loops on flat regions
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                
                if fr < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Just replace worst
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    # Outside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < f_values[-1]:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for k in range(1, n + 1):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            f_values[k] = func(simplex[k])
                
                # Sort simplex again
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break
                
                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
