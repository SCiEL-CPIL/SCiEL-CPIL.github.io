import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy):
    """Return vertices of a unit hexagon centered at (cx, cy) with angle 0."""
    # Angle 0 means flat top? 
    # Standard math: vertex at (1,0).
    # Task says "angle_degrees". 
    # If we use angle 0 for all, we assume they are aligned.
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    angles = np.arange(0, 360, 60) * np.pi / 180
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def _calc_outer_side(centers):
    """Calculate minimum side length S of outer hexagon (angle 0) to contain all inner hexagons."""
    max_s = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy)
        for x, y in verts:
            # Constraints for outer hexagon of side S, angle 0
            # |x| <= S
            # |sqrt(3)*x + y| <= sqrt(3)*S
            # |-sqrt(3)*x + y| <= sqrt(3)*S
            
            s1 = abs(x)
            s2 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
            s3 = abs(-np.sqrt(3)*x + y) / np.sqrt(3)
            
            local_s = max(s1, s2, s3)
            if local_s > max_s:
                max_s = local_s
    return max_s

def _overlap_penalty(centers, threshold=np.sqrt(3) - 1e-6):
    """Calculate penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < threshold:
                penalty += (threshold - dist)**2
    return penalty

def _objective(params):
    """Objective function for optimization."""
    centers = params.reshape(-1, 2)
    s = _calc_outer_side(centers)
    pen = _overlap_penalty(centers)
    return s + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate honeycomb initial configuration
    # Grid coordinates (q, r)
    grid_points = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,2), (-2,1), (-1,-1), (2,-1)
    ]
    
    centers = []
    for q, r in grid_points:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the configuration at origin
    centers[:, 0] -= np.mean(centers[:, 0])
    centers[:, 1] -= np.mean(centers[:, 1])
    
    # Initial evaluation
    best_params = centers.flatten()
    best_score = _calc_outer_side(centers)
    
    # 2. Optimize using L-BFGS-B
    # We want to minimize the outer side length.
    # The overlap penalty ensures validity.
    
    try:
        result = minimize(
            _objective, 
            best_params, 
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000}
        )
        
        if time.time() < deadline:
            opt_centers = result.x.reshape(-1, 2)
            opt_s = _calc_outer_side(opt_centers)
            opt_pen = _overlap_penalty(opt_centers)
            
            # Check if the optimized solution is valid (low penalty) and better
            if opt_pen < 1e-4 and opt_s < best_score:
                best_params = result.x
                best_score = opt_s
                centers = opt_centers
                
    except Exception:
        pass

    # Ensure we return the best valid configuration found
    final_centers = best_params.reshape(-1, 2)
    final_s = _calc_outer_side(final_centers)
    
    # If the solution is invalid (overlap), we might need to fallback to a known valid one
    # But L-BFGS with penalty usually keeps them valid if started valid.
    # Let's double check validity.
    if _overlap_penalty(final_centers) > 1e-4:
        # Fallback to initial honeycomb
        final_centers = centers
        final_s = _calc_outer_side(final_centers)

    # Construct return format
    # inner: list of [x, y, angle]
    inner = []
    for x, y in final_centers:
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon
    # Center (0,0), side final_s, angle 0
    return inner, (0.0, 0.0), float(final_s), 0.0

# EVOLVE-BLOCK-END
