from scipy.optimize import minimize
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lattice coordinates for 11 hexagons
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (2, -2), (1, -2)
    ]
    
    # Convert to Cartesian
    # Apply a small scaling factor to ensure strict disjointness (gap > 0)
    # This prevents floating point "overlap" errors for touching hexagons
    scale_factor = 1.001 
    centers_rel = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0) * scale_factor
        y = 1.5 * r * scale_factor
        centers_rel.append((x, y))
        
    # Pre-compute vertices
    base_angles = np.arange(6) * np.pi / 3
    base_vx = np.cos(base_angles)
    base_vy = np.sin(base_angles)
    
    V_rel = np.zeros((11, 6, 2))
    for i, (cx, cy) in enumerate(centers_rel):
        V_rel[i, :, 0] = cx + base_vx
        V_rel[i, :, 1] = cy + base_vy
        
    verts_flat = V_rel.reshape(66, 2)
    
    # Normals for outer hexagon (fixed orientation)
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    apothem = np.sqrt(3) / 2
    
    def fast_objective(params):
        cx, cy, alpha = params
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        
        rx = verts_flat[:, 0] * cos_a - verts_flat[:, 1] * sin_a + cx
        ry = verts_flat[:, 0] * sin_a + verts_flat[:, 1] * cos_a + cy
        
        proj = rx[np.newaxis, :] * nx[:, np.newaxis] + ry[np.newaxis, :] * ny[:, np.newaxis]
        max_proj = np.max(np.abs(proj))
        
        return max_proj / apothem
    
    p0 = [0.0, 0.0, 0.0]
    
    # Optimization
    best_S = float('inf')
    best_params = p0
    
    # Run a few iterations or just one robust run
    try:
        result = minimize(fast_objective, p0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < best_S:
            best_S = result.fun
            best_params = result.x
            
        # Check time
        if time.time() > deadline:
            pass # Use best so far
            
        # Try a few random rotations to escape local minima if time permits
        # The objective is periodic and likely has symmetries, so 0 is often good,
        # but let's try one more start if fast.
        if time.time() < deadline - 5:
            p0_alt = [0.0, 0.0, np.pi / 6] # 30 deg offset
            result2 = minimize(fast_objective, p0_alt, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result2.fun < best_S:
                best_S = result2.fun
                best_params = result2.x
                
    except Exception as e:
        # Fallback to initial guess if optimization fails
        best_S = fast_objective(p0)
        best_params = p0

    cx, cy, alpha = best_params
    
    inner = []
    for i, (rel_x, rel_y) in enumerate(centers_rel):
        g_x = rel_x * np.cos(alpha) - rel_y * np.sin(alpha) + cx
        g_y = rel_x * np.sin(alpha) + rel_y * np.cos(alpha) + cy
        inner.append([g_x, g_y, np.degrees(alpha)])
        
    return inner, (0.0, 0.0), best_S, 0.0
