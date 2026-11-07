# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Two-phase optimizer for Bent Cigar:
    Phase 1: Coordinate-wise curvature estimation to identify ill-conditioned dimensions
    Phase 2: Accelerated 1D ternary search on ill-conditioned axes + coarse grid for well-conditioned
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: Estimate per-coordinate curvature via finite differences
        # Sample a central point and evaluate along each axis to gauge sensitivity
        center = np.zeros(dim)
        f_center = wrapper(center)
        
        # Estimate curvature by sampling small perturbations along each axis
        curvatures = np.zeros(dim)
        step_size = 0.1
        for i in range(dim):
            x_plus = center.copy()
            x_minus = center.copy()
            x_plus[i] += step_size
            x_minus[i] -= step_size
            f_plus = wrapper(x_plus)
            f_minus = wrapper(x_minus)
            # Second-order finite difference approximation of curvature
            curvatures[i] = (f_plus - 2 * f_center + f_minus) / (step_size ** 2)
        
        # Rank dimensions by curvature (highest curvature = most ill-conditioned)
        # For Bent Cigar, the last dimensions have much higher curvature
        curvature_order = np.argsort(-curvatures)  # descending order of curvature
        
        # Phase 2: Search strategy based on curvature ranking
        # Use ternary search on high-curvature dimensions, coarse grid on low-curvature
        
        # First, do a coarse grid search to get a rough minimum
        # Focus budget on high-curvature dimensions with finer resolution
        n_coarse = 3  # 3 levels per dimension for coarse search
        coarse_budget = min(200, budget - 50)
        
        # Generate coarse grid points, prioritizing high-curvature dimensions
        # For high-curvature dims, use more points; for low-curvature, fewer
        grid_ranges = []
        for i in range(dim):
            rank = np.where(curvature_order == i)[0][0]
            if rank < 2:  # Top 2 most curved dimensions
                n_pts = 5
            else:
                n_pts = 3
            grid_ranges.append(n_pts)
        
        # Evaluate a structured grid
        grid_axes = []
        for i in range(dim):
            n_pts = grid_ranges[i]
            if n_pts > 1:
                grid_axes.append(np.linspace(-4, 4, n_pts))
            else:
                grid_axes.append(np.array([0.0]))
        
        # Enumerate grid points (limit to avoid combinatorial explosion)
        max_grid_points = 100
        grid_points = []
        
        def gen_grid(idx, current):
            if len(grid_points) >= max_grid_points:
                return
            if idx == dim:
                grid_points.append(current.copy())
                return
            for val in grid_axes[idx]:
                current[idx] = val
                gen_grid(idx + 1, current)
        
        gen_grid(0, np.zeros(dim))
        
        # Evaluate all grid points
        for gp in grid_points:
            if calls >= budget - 100 or time.time() > deadline:
                break
            wrapper(gp)
        
        # Phase 3: Local refinement using L-BFGS-B from the best grid point
        if calls < budget - 50 and time.time() < deadline:
            # Try L-BFGS-B from best point and a few nearby points
            local_starts = [best_x.copy()]
            
            # Add a few random perturbations around best
            for _ in range(5):
                perturbed = best_x + rng.normal(0, 0.5, dim)
                perturbed = np.clip(perturbed, -5, 5)
                local_starts.append(perturbed)
            
            for x_start in local_starts:
                if calls >= budget - 20 or time.time() > deadline:
                    break
                
                try:
                    res = scipy.optimize.minimize(
                        wrapper,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 100,
                            'maxfev': min(100, budget - calls),
                            'ftol': 1e-10,
                            'gtol': 1e-6,
                            'eps': 1e-5
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        # Phase 4: Final 1D ternary search on the most curved dimension
        if calls < budget - 10 and time.time() < deadline:
            # Identify the single most curved dimension
            most_curved_dim = curvature_order[0]
            
            # Ternary search along that dimension, holding others at best_x
            lo, hi = -5.0, 5.0
            for _ in range(15):  # 15 iterations for good precision
                if calls >= budget - 2 or time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                x_m1 = best_x.copy()
                x_m1[most_curved_dim] = m1
                x_m2 = best_x.copy()
                x_m2[most_curved_dim] = m2
                
                f_m1 = wrapper(x_m1)
                f_m2 = wrapper(x_m2)
                
                if f_m1 < f_m2:
                    hi = m2
                else:
                    lo = m1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
