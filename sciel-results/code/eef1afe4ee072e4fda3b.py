import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    # 1. Initialize positions in a compact hexagonal cluster
    # For N=11, a 3x3 grid in hex coordinates works well, or just a circle.
    # Let's use a simple hexagonal grid layout for initialization.
    # Hex grid coordinates: (q, r) -> (x, y)
    # x = size * (sqrt(3)*q + sqrt(3)/2*r)
    # y = size * (3/2*r)
    # For unit side length hexagons, distance between centers is sqrt(3).
    
    # Generate hex grid points around origin
    points = []
    radius = 2 # covers 1 + 6 + 12 = 19 points, we take first 11
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                x = np.sqrt(3) * (q + 0.5 * r)
                y = 1.5 * r
                points.append([x, y])
    
    # Sort by distance from origin to get the most compact cluster
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_pos = np.array(points[:N])
    
    # 2. Define cost function
    # We want to minimize the outer hexagon side length L.
    # For a set of points, the minimum enclosing regular hexagon (with fixed orientation)
    # has side length L such that all points are inside.
    # A point (x,y) is inside a regular hexagon of side L centered at origin if:
    # |x| <= L*sqrt(3)/2  AND  |y| <= L  AND  |x/sqrt(3) + y/2| <= L  AND  |-x/sqrt(3) + y/2| <= L
    # Actually, the support function of a regular hexagon of side L in direction u is L * sqrt(3)/2 * |cos(theta)| + ... 
    # Simpler: The distance from center to side is apothem a = L * sqrt(3)/2.
    # The hexagon is defined by 6 inequalities: n_i . p <= a for 6 normal vectors.
    # To minimize L, we minimize the maximum of (n_i . p) / a_i where a_i depends on L.
    # This is circular. 
    # Alternative: Minimize the maximum projection onto the 6 normal directions.
    # Let proj_i = max_j (n_i . p_j).
    # Then L must satisfy proj_i <= L * sqrt(3)/2 for all i.
    # So L_min = max_i (proj_i) * 2 / sqrt(3).
    
    # Normals for regular hexagon (flat top/bottom vs pointy top)
    # Let's assume flat top (sides horizontal).
    # Normals are at 0, 60, 120, 180, 240, 300 degrees? No, normals to sides.
    # If sides are horizontal, normals are vertical (90, 270) and slanted (30, 150, 210, 330).
    # Let's use pointy top (vertices at 0, 60, ...). Sides are vertical? No.
    # Standard orientation: Vertices at (L, 0), (L/2, L*sqrt(3)/2)...
    # Normals to sides are at 30, 90, 150, 210, 270, 330 degrees.
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    def calc_cost(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        
        # Check overlaps
        # Distance between centers must be >= sqrt(3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Penalty for overlap
        # If min_dist < sqrt(3), we have overlap.
        # Penalty should be large.
        overlap_penalty = 0.0
        if min_dist < np.sqrt(3):
            # Sum of violations squared to smooth
            violations = np.maximum(0, np.sqrt(3) - dists)
            overlap_penalty = 100.0 * np.sum(violations**2)
        
        # Calculate L
        # Project points onto 6 normal directions
        proj = pos @ np.column_stack((cos_a, sin_a))
        # For each direction, we need the max positive projection and min negative projection?
        # The hexagon is centered at (0,0). 
        # Condition: |n_i . p| <= apothem = L * sqrt(3)/2
        # So L >= max_i (max_j |n_i . p_j|) * 2 / sqrt(3)
        
        abs_proj = np.abs(proj)
        max_abs_proj = np.max(abs_proj)
        L = max_abs_proj * 2.0 / np.sqrt(3)
        
        return L + overlap_penalty

    # 3. Optimize
    best_pos = initial_pos
    best_L = np.inf
    
    # Nelder-Mead is good for this low-dimensional problem
    # We allow translation of the whole cluster to center it? 
    # The cost function assumes center is (0,0). 
    # We should also optimize the center position.
    # Total variables: 2*N + 2 (centers x,y)
    
    def full_cost(params):
        pos = params[:2*N].reshape(-1, 2)
        center = params[2*N:2*N+2]
        pos_centered = pos - center
        
        # Overlaps
        diff = pos_centered[:, np.newaxis, :] - pos_centered[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        overlap_penalty = 0.0
        if min_dist < np.sqrt(3):
            violations = np.maximum(0, np.sqrt(3) - dists)
            overlap_penalty = 100.0 * np.sum(violations**2)
        
        # L calculation
        proj = pos_centered @ np.column_stack((cos_a, sin_a))
        max_abs_proj = np.max(np.abs(proj))
        L = max_abs_proj * 2.0 / np.sqrt(3)
        
        return L + overlap_penalty

    # Initial params: positions + center (0,0)
    init_params = np.concatenate([initial_pos.flatten(), [0.0, 0.0]])
    
    # Run optimization
    try:
        result = minimize(full_cost, init_params, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.success or result.fun < best_L:
            best_params = result.x
            best_pos = best_params[:2*N].reshape(-1, 2)
            best_center = best_params[2*N:2*N+2]
            
            # Verify validity
            pos_c = best_pos - best_center
            diff = pos_c[:, np.newaxis, :] - pos_c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= np.sqrt(3) - 1e-6:
                # Calculate final L
                proj = pos_c @ np.column_stack((cos_a, sin_a))
                L = np.max(np.abs(proj)) * 2.0 / np.sqrt(3)
                best_L = L
            else:
                # Fallback: Use initial config if optimization failed to find valid
                # Initial config is a hex grid, which is valid.
                best_pos = initial_pos
                best_center = np.array([0.0, 0.0])
                proj = best_pos @ np.column_stack((cos_a, sin_a))
                L = np.max(np.abs(proj)) * 2.0 / np.sqrt(3)
                best_L = L

    except Exception:
        best_pos = initial_pos
        best_center = np.array([0.0, 0.0])
        proj = best_pos @ np.column_stack((cos_a, sin_a))
        L = np.max(np.abs(proj)) * 2.0 / np.sqrt(3)
        best_L = L

    # 4. Format output
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_L), 0.0
