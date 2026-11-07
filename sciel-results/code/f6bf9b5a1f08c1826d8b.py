import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = (-5.0, 5.0)
        
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin under 330s
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def track(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, bounds[0], bounds[1])
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid search to find a promising basin
        # Use a denser grid near the origin since minimizer is in [-4,4]^5
        levels = [-4.0, -2.0, 0.0, 2.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=dim))
        
        # Sort by distance to origin for early progress
        origin = np.zeros(dim)
        grid_points.sort(key=lambda p: np.linalg.norm(np.array(p) - origin))
        
        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            try:
                track(np.array(p, dtype=np.float64))
            except Exception:
                break

        # Phase 2: Coordinate descent with golden-section search
        # This is a structural change: instead of Nelder-Mead, use 1D golden-section
        # search along each coordinate, which is more sample-efficient for smooth functions
        
        def golden_section_1d(x, coord_idx, left, right, max_evals=20):
            """Perform golden-section search along one coordinate."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return x
            
            golden_ratio = (1 + np.sqrt(5)) / 2
            inv_phi = 1.0 / golden_ratio
            inv_phi_sq = inv_phi * inv_phi
            
            a = left
            b = right
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            # Evaluate initial points
            x_c = x.copy()
            x_c[coord_idx] = c
            x_d = x.copy()
            x_d[coord_idx] = d
            
            try:
                f_c = track(x_c)
                f_d = track(x_d)
            except Exception:
                return x
            
            for _ in range(max_evals):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_phi * (b - a)
                    x_c = x.copy()
                    x_c[coord_idx] = c
                    try:
                        f_c = track(x_c)
                    except Exception:
                        break
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_phi * (b - a)
                    x_d = x.copy()
                    x_d[coord_idx] = d
                    try:
                        f_d = track(x_d)
                    except Exception:
                        break
                
                # Check convergence
                if abs(b - a) < 1e-8:
                    break
            
            # Return best point found
            mid = (a + b) / 2
            x_best = x.copy()
            x_best[coord_idx] = mid
            return x_best

        def coordinate_descent(x0, max_rounds=10, gs_evals=15):
            """Perform coordinate descent with golden-section search."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return x0
            
            x = x0.copy()
            rng = np.random.default_rng(42)
            
            for round_num in range(max_rounds):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Shuffle coordinates to avoid bias
                coords = list(range(dim))
                rng.shuffle(coords)
                
                improved = False
                for coord_idx in coords:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Determine search range for this coordinate
                    # Use a window around current value
                    center = x[coord_idx]
                    half_width = 2.0  # Search in [center-2, center+2]
                    left = max(bounds[0], center - half_width)
                    right = min(bounds[1], center + half_width)
                    
                    x_new = golden_section_1d(x, coord_idx, left, right, gs_evals)
                    
                    if np.any(x_new != x):
                        x = x_new
                        improved = True
                
                if not improved:
                    break
            
            return x

        # Run coordinate descent from the best point found in grid search
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            # Use most of remaining budget for coordinate descent
            cd_rounds = min(15, max(5, remaining // 30))
            gs_evals = min(20, max(10, remaining // 20))
            
            try:
                refined = coordinate_descent(best_x, cd_rounds, gs_evals)
                track(refined)
            except Exception:
                pass

        # Phase 3: Try a few more restarts if budget allows
        if calls_made < budget - 20 and time.time() < deadline:
            rng = np.random.default_rng(123)
            for _ in range(3):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                x0 = rng.uniform(-3, 3, dim)
                try:
                    refined = coordinate_descent(x0, 5, 10)
                    track(refined)
                except Exception:
                    break

        return best_x, best_f
