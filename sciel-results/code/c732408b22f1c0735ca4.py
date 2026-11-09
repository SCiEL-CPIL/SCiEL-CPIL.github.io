import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Uses a differentiable soft-constraint loss with smooth overlap penalty.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    def check_validity(s, cx, cy):
        """Check if the configuration is strictly valid."""
        tol = 1e-7
        if not np.all(np.isfinite([s, cx, cy])):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def objective(params):
        """
        Objective function for local optimization.
        Returns -sum(sides) + penalty terms.
        Uses smooth hinge for overlap to make the loss differentiable.
        """
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary constraints (smooth)
        for i in range(n):
            if s[i] <= 0:
                penalty += 10.0 * (1.0 - s[i])**2
                continue
            
            # X boundaries
            d_left = s[i]/2 - cx[i]
            d_right = cx[i] + s[i]/2 - 1.0
            if d_left > 0:
                penalty += d_left**2
            if d_right > 0:
                penalty += d_right**2
            
            # Y boundaries
            d_bottom = s[i]/2 - cy[i]
            d_top = cy[i] + s[i]/2 - 1.0
            if d_bottom > 0:
                penalty += d_bottom**2
            if d_top > 0:
                penalty += d_top**2
        
        # Overlap constraints (smooth)
        eps = 1e-6
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2.0
                sy = (s[i] + s[j]) / 2.0
                
                ox = sx - dx
                oy = sy - dy
                
                # Smooth hinge for overlap
                smooth_ox = 0.5 * (ox + np.sqrt(ox*ox + eps))
                smooth_oy = 0.5 * (oy + np.sqrt(oy*oy + eps))
                
                penalty += 10.0 * smooth_ox * smooth_oy
        
        return obj + penalty

    def local_optimize(params, maxiter=1000, ftol=1e-9, gtol=1e-6):
        """Run L-BFGS-B on a given starting point."""
        bounds = []
        for _ in range(n):
            bounds.append((1e-6, 1.0)) # s
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cy
            
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            return res.x
        except:
            return params

    best_params = None
    best_score = -np.inf

    def update_best(params):
        nonlocal best_params, best_score
        if params is None:
            return
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        if check_validity(s, cx, cy):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_params = params.copy()

    # --- Strategy 1: Deterministic Grid Initializations ---
    
    # Init 1: 4x3 Grid (Sum = 2.5)
    k = 4
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side
    cy_grid = (rows + 0.5) * side
    s_grid = np.full(n, side)
    params_grid = np.concatenate([s_grid, cx_grid, cy_grid])
    update_best(params_grid)

    # Init 2: Hierarchical (1 large, 9 small) - Known good structure (Sum = 3.0)
    # 2 squares of 0.5 and 8 squares of 0.25
    s_hier = np.zeros(n)
    cx_hier = np.zeros(n)
    cy_hier = np.zeros(n)
    
    # Two big squares
    s_hier[0] = 0.5
    cx_hier[0] = 0.25
    cy_hier[0] = 0.25
    
    s_hier[1] = 0.5
    cx_hier[1] = 0.75
    cy_hier[1] = 0.75
    
    # 8 small squares
    # Region 1: x in [0.5, 1], y in [0, 0.5]
    for i in range(4):
        r = i // 2
        c = i % 2
        s_hier[2+i] = 0.25
        cx_hier[2+i] = 0.625 + c * 0.25
        cy_hier[2+i] = 0.125 + r * 0.25
        
    # Region 2: x in [0, 0.5], y in [0.5, 1]
    for i in range(4):
        r = i // 2
        c = i % 2
        s_hier[6+i] = 0.25
        cx_hier[6+i] = 0.125 + c * 0.25
        cy_hier[6+i] = 0.625 + r * 0.25
        
    params_hier = np.concatenate([s_hier, cx_hier, cy_hier])
    update_best(params_hier)

    # Init 3: Strip layout - 5 squares in bottom row, 5 in top row
    s_strip = np.zeros(n)
    cx_strip = np.zeros(n)
    cy_strip = np.zeros(n)
    side_strip = 1.0/5
    for i in range(5):
        s_strip[i] = side_strip
        cx_strip[i] = (i + 0.5) * side_strip
        cy_strip[i] = 0.25 * side_strip
        s_strip[5+i] = side_strip
        cx_strip[5+i] = (i + 0.5) * side_strip
        cy_strip[5+i] = 0.75 * side_strip
    params_strip = np.concatenate([s_strip, cx_strip, cy_strip])
    update_best(params_strip)

    # --- Strategy 2: Local Optimization on Initializations ---
    
    if time.time() < deadline - 10:
        polished_hier = local_optimize(params_hier, maxiter=2000)
        update_best(polished_hier)
        
        polished_grid = local_optimize(params_grid, maxiter=2000)
        update_best(polished_grid)
        
        polished_strip = local_optimize(params_strip, maxiter=2000)
        update_best(polished_strip)

    # --- Strategy 3: More Diverse Initializations ---
    # Try random restarts with local optimization
    
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        
        # Create a random but somewhat structured initialization
        # Mix of large and small squares
        s_rand = np.random.uniform(0.1, 0.4, n)
        cx_rand = np.random.uniform(0, 1, n)
        cy_rand = np.random.uniform(0, 1, n)
        params_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        
        # Quick local optimization
        opt_params = local_optimize(params_rand, maxiter=500)
        update_best(opt_params)

    # --- Strategy 4: Perturbation of Best Solution ---
    if best_params is not None and time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution found so far
            noise_s = np.random.normal(0, 0.02, n)
            noise_cx = np.random.normal(0, 0.02, n)
            noise_cy = np.random.normal(0, 0.02, n)
            
            perturbed_s = np.clip(best_params[:n] + noise_s, 1e-6, 1.0)
            perturbed_cx = np.clip(best_params[n:2*n] + noise_cx, 0.0, 1.0)
            perturbed_cy = np.clip(best_params[2*n:3*n] + noise_cy, 0.0, 1.0)
            
            perturbed_params = np.concatenate([perturbed_s, perturbed_cx, perturbed_cy])
            
            opt_params = local_optimize(perturbed_params, maxiter=1000)
            update_best(opt_params)

    # --- Final Polish with tighter tolerances ---
    if best_params is not None and time.time() < deadline - 5:
        # First pass with standard settings
        polished_final = local_optimize(best_params, maxiter=5000)
        update_best(polished_final)
        
        # Second pass with tighter tolerances for high-precision polish
        if time.time() < deadline - 3:
            polished_tight = local_optimize(polished_final, maxiter=10000, ftol=1e-12, gtol=1e-8)
            update_best(polished_tight)

    # Fallback
    if best_params is None:
        centers = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
        sides = s_grid.astype(np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
