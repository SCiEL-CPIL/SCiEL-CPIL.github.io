import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Uses a differentiable soft-constraint loss with smooth maximum for overlap,
    allowing gradient-based solvers (L-BFGS-B) to directly maximize the objective.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

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

    def smooth_max(x, y, k=10.0):
        """Smooth approximation of max(x, y)."""
        return (x + y + np.sqrt((x - y)**2 + 1.0/k**2)) / 2.0

    def objective(params):
        """
        Objective function for local optimization.
        Returns -sum(sides) + penalty terms.
        Uses smooth maximum for overlap to make the loss differentiable.
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
        # We want to penalize when squares overlap.
        # Overlap in x: max(0, (s_i+s_j)/2 - |cx_i-cx_j|)
        # Overlap in y: max(0, (s_i+s_j)/2 - |cy_i-cy_j|)
        # If both > 0, they overlap.
        # To make it smooth, we use smooth_max for the positive part.
        # Actually, a common trick is to penalize the product of overlaps if we assume they can't be negative?
        # No, overlap amount is max(0, ...).
        # Let's use a smooth hinge: smooth_hinge(x) = 0.5 * (x + sqrt(x^2 + eps))
        # This is smooth and approximates max(0, x).
        
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
                # We want penalty when ox > 0 AND oy > 0.
                # A simple product of smooth hinges works well.
                smooth_ox = 0.5 * (ox + np.sqrt(ox*ox + eps))
                smooth_oy = 0.5 * (oy + np.sqrt(oy*oy + eps))
                
                # Penalty is proportional to the area of overlap (approx)
                # Or just the product to ensure it's zero if either is ~0
                penalty += 10.0 * smooth_ox * smooth_oy
                
        return obj + penalty

    def local_optimize(params, maxiter=1000):
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
                options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6}
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

    # --- Strategy 2: Local Optimization on Initializations ---
    
    if time.time() < deadline - 10:
        polished_hier = local_optimize(params_hier, maxiter=2000)
        update_best(polished_hier)
        
        polished_grid = local_optimize(params_grid, maxiter=2000)
        update_best(polished_grid)

    # --- Strategy 3: More Diverse Initializations ---
    # Try random restarts with local optimization
    
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        
        # Create a random but somewhat structured initialization
        # Mix of large and small squares
        s_rand = np.random.uniform(0.1, 0.4, n)
        # Sort sides to try to place larger ones first? Or just random positions
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

    # --- Final Polish ---
    if best_params is not None and time.time() < deadline - 5:
        polished_final = local_optimize(best_params, maxiter=5000)
        update_best(polished_final)

    # Fallback
    if best_params is None:
        centers = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
        sides = s_grid.astype(np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
