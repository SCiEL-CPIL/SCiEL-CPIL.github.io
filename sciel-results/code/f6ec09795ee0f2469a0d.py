# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0
    
    # Baseline: 3x3 grid (9 squares) + 1 extra
    # Better baseline: mixed sizes
    # Try a layout with 4 larger squares in corners and 6 smaller ones
    # Or use a more optimized initial layout
    
    # Initial best: simple grid
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    centers_base = np.stack([cx_base, cy_base], axis=1).astype(np.float64)
    sides_base = np.full(n, side, dtype=np.float64)
    best_score = np.sum(sides_base)
    best_centers = centers_base.copy()
    best_sides = sides_base.copy()

    def unpack(params):
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        return cx, cy, s

    def penalty_objective(params, penalty_weight=1e6):
        """Smooth penalty-based objective for optimization"""
        cx, cy, s = unpack(params)
        
        # Boundary violations
        penalty = 0.0
        for i in range(n):
            left_viol = max(0, -1e-7 - (cx[i] - s[i]/2))
            right_viol = max(0, (cx[i] + s[i]/2) - (1 + 1e-7))
            bottom_viol = max(0, -1e-7 - (cy[i] - s[i]/2))
            top_viol = max(0, (cy[i] + s[i]/2) - (1 + 1e-7))
            penalty += (left_viol + right_viol + bottom_viol + top_viol) ** 2
        
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                # Overlap if both dx < t and dy < t
                # Penalty for each violated dimension
                x_overlap = max(0, t - 1e-7 - dx)
                y_overlap = max(0, t - 1e-7 - dy)
                if x_overlap > 0 and y_overlap > 0:
                    penalty += (x_overlap * y_overlap) ** 2
        
        return -np.sum(s) + penalty_weight * penalty

    def hard_objective(params):
        """Hard constraint objective for final validation"""
        cx, cy, s = unpack(params)
        
        for i in range(n):
            if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                return 1e10
            if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                return 1e10
        
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - 1e-7 and dy < t - 1e-7:
                    return 1e10
        
        return -np.sum(s)

    def is_valid(params):
        """Check if a solution is valid"""
        cx, cy, s = unpack(params)
        if not np.all(np.isfinite(params)):
            return False
        if not np.all(s > 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                return False
            if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - 1e-7 and dy < t - 1e-7:
                    return False
        return True

    def update_best(params):
        nonlocal best_score, best_centers, best_sides
        if is_valid(params):
            cx, cy, s = unpack(params)
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
                best_sides = s.astype(np.float64)
                return True
        return False

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s

    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Uniform grid (baseline)
    initial_configs.append(np.hstack([cx_base, cy_base, sides_base]))
    
    # Config 2: 4 large squares in corners, 6 smaller filling middle
    # Large squares: side ~0.35, small: side ~0.2
    s_large = 0.38
    s_small = 0.22
    config2 = np.zeros(3*n)
    # 4 large in corners
    config2[0] = s_large/2; config2[n] = s_large/2; config2[2*n] = s_large
    config2[1] = 1-s_large/2; config2[n+1] = s_large/2; config2[2*n+1] = s_large
    config2[2] = s_large/2; config2[n+2] = 1-s_large/2; config2[2*n+2] = s_large
    config2[3] = 1-s_large/2; config2[n+3] = 1-s_large/2; config2[2*n+3] = s_large
    # 6 smaller in middle area
    remaining = 6
    for i in range(remaining):
        idx = 4 + i
        config2[idx] = 0.5
        config2[n+idx] = 0.5
        config2[2*n+idx] = s_small
    # Spread them out
    positions = [(0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65), (0.5, 0.3), (0.5, 0.7)]
    for i in range(min(remaining, len(positions))):
        idx = 4 + i
        config2[idx] = positions[i][0]
        config2[n+idx] = positions[i][1]
    initial_configs.append(config2)
    
    # Config 3: Mixed - 2 large, 8 medium
    s_l = 0.45
    s_m = 0.25
    config3 = np.zeros(3*n)
    config3[0] = s_l/2; config3[n] = s_l/2; config3[2*n] = s_l
    config3[1] = 1-s_l/2; config3[n+1] = 1-s_l/2; config3[2*n+1] = s_l
    for i in range(8):
        idx = 2 + i
        config3[idx] = 0.5 + 0.3*np.cos(2*np.pi*i/8)
        config3[n+idx] = 0.5 + 0.3*np.sin(2*np.pi*i/8)
        config3[2*n+idx] = s_m
    initial_configs.append(config3)

    # Run differential evolution from multiple starts
    elapsed = time.time() - start_time
    remaining_time = deadline - elapsed - safety_margin
    
    if remaining_time > 60:
        # Use penalty objective for DE (smoother landscape)
        try:
            de_result = differential_evolution(
                penalty_objective, 
                bounds, 
                maxiter=150, 
                popsize=25, 
                seed=42,
                tol=0.01,
                init='sobol',
                polish=False
            )
            update_best(de_result.x)
        except Exception as e:
            pass
        
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed - safety_margin

    # Local refinement from multiple starting points
    if remaining_time > 30:
        start_points = [best_centers[:,0].copy(), best_centers[:,1].copy(), best_sides.copy()]
        start_points = [np.hstack([start_points[0], start_points[1], start_points[2]])]
        
        # Add perturbed versions
        for _ in range(5):
            perturbation = np.random.normal(0, 0.05, 3*n)
            sp = best_centers[:,0].copy() + perturbation[0:n]
            sp_cy = best_centers[:,1].copy() + perturbation[n:2*n]
            sp_s = best_sides.copy() + perturbation[2*n:3*n]
            # Clip to valid ranges
            sp = np.clip(sp, 0, 1)
            sp_cy = np.clip(sp_cy, 0, 1)
            sp_s = np.clip(sp_s, 0.01, 1)
            start_points.append(np.hstack([sp, sp_cy, sp_s]))
        
        # Also add the initial configs
        for cfg in initial_configs:
            start_points.append(cfg.copy())
        
        for sp in start_points:
            if time.time() - start_time > deadline - safety_margin:
                break
            try:
                res_nm = minimize(
                    penalty_objective, 
                    sp, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                update_best(res_nm.x)
            except Exception as e:
                pass

    # Final validation and return
    final_params = np.hstack([best_centers[:,0], best_centers[:,1], best_sides])
    if is_valid(final_params):
        return best_centers, best_sides
    else:
        # Fallback to baseline if somehow invalid
        return centers_base, sides_base

# EVOLVE-BLOCK-END
