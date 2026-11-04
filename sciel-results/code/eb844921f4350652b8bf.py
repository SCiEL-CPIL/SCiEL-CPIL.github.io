import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    # Lattice vectors for honeycomb packing of unit hexagons
    # Distance between centers is sqrt(3)
    # We use a small margin to ensure strict disjointness if needed, 
    # but for a valid packing, exact touching is fine if the evaluator allows it.
    # The prompt says "disjoint", which usually means non-overlapping interiors.
    # Touching is allowed.
    scale = 1.0
    u = np.array([np.sqrt(3), 0.0]) * scale
    v = np.array([np.sqrt(3)/2.0, 1.5]) * scale
    
    points = []
    for i in range(4):
        for j in range(3):
            p = i * u + j * v
            points.append(p)
            
    points = np.array(points)
    
    # Center the points around the origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Pre-compute vertices of a single unit hexagon centered at origin
    # Standard orientation: flat top/bottom? Or pointy top?
    # Let's assume "regular hexagon" implies vertices at angles 0, 60, 120, 180, 240, 300
    # if rotation is 0. 
    # If we want flat sides on x-axis, rotation is 30.
    # Let's define local vertices for a unit hexagon.
    # Side length 1. Distance from center to vertex is 1.
    local_vertices = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Function to get global vertices for all inner hexagons
    def get_global_vertices(inner_rotation_deg):
        # inner_rotation_deg is the rotation of the inner hexagons
        # We assume all inner hexagons have the same rotation for simplicity in this baseline.
        # The lattice is symmetric, so this is a good guess.
        theta = np.deg2rad(inner_rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Transform local vertices
        rotated_local = local_vertices @ R.T
        
        # Global vertices: (points[:, i] + rotated_local[j])
        # We have 12 points and 6 vertices.
        # Result shape: (12, 6, 2)
        global_verts = points[:, np.newaxis, :] + rotated_local[np.newaxis, :, :]
        return global_verts

    # Function to compute the required side length S for a given outer rotation phi and center C
    # given a specific inner rotation
    def objective(params, inner_rot):
        phi, cx, cy = params
        C = np.array([cx, cy])
        
        # Get all global vertices of all inner hexagons
        # This is expensive if done inside the loop. 
        # However, inner_rot is fixed during the optimization of phi/cx/cy.
        # So we can precompute global vertices for the current inner_rot outside or cache it.
        # For simplicity and correctness first, we compute it here. 
        # Optimization: We can precompute for a fixed inner_rot.
        
        g_verts = get_global_vertices(inner_rot)
        # Flatten to (72, 2)
        g_verts_flat = g_verts.reshape(-1, 2)
        
        max_proj = 0.0
        for k in range(6):
            angle = phi + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            # Project all vertices onto n
            projs = np.dot(g_verts_flat - C, n)
            # The max absolute projection is the required apothem in this direction
            max_abs_proj = np.max(np.abs(projs))
            if max_abs_proj > max_proj:
                max_proj = max_abs_proj
        
        # Apothem a = max_proj
        # Side length S = 2 * a / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)
        
    # We need to optimize:
    # 1. Inner rotation (likely 0 or 30 or 60)
    # 2. Outer rotation
    # 3. Outer center
    
    best_side = float('inf')
    best_params = None
    
    # Try a few inner rotations
    # The honeycomb lattice has 6-fold symmetry. 
    # Aligning inner hexagon sides with outer hexagon sides (0 deg relative) or vertices with sides (30 deg) are candidates.
    inner_rot_candidates = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    for ir in inner_rot_candidates:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Define objective for this specific inner rotation
        def obj_wrapper(params):
            return objective(params, ir)
        
        # Initial guess for outer rotation and center
        # Due to symmetry, center is likely (0,0)
        # Outer rotation can be anything, but let's scan or optimize.
        # Let's use Nelder-Mead to optimize phi, cx, cy
        x0 = [0.0, 0.0, 0.0]
        
        # To handle the non-smoothness of max(), we can try multiple starts for phi
        best_local = None
        for phi_start in np.linspace(0, np.pi/6, 5):
            x0 = [phi_start, 0.0, 0.0]
            try:
                res = minimize(obj_wrapper, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 100})
                if best_local is None or res.fun < best_local.fun:
                    best_local = res
            except Exception:
                pass
        
        if best_local is not None and best_local.fun < best_side:
            best_side = best_local.fun
            best_params = (best_local.x, ir)
            
    if best_params is None:
        # Fallback: Just return a large hexagon to be valid
        # Calculate bounding box of the lattice with 0 rotation
        g_verts = get_global_vertices(0.0).reshape(-1, 2)
        max_x = np.max(np.abs(g_verts[:, 0]))
        max_y = np.max(np.abs(g_verts[:, 1]))
        # Rough estimate
        side = 10.0
        return [[p[0], p[1], 0.0] for p in points], [0, 0], side, 0.0

    (phi_opt, cx_opt, cy_opt), ir_opt = best_params
    S_opt = best_side
    
    # Ensure the result is valid by slightly inflating the side length?
    # The evaluator checks containment. If we hit the boundary exactly, floating point errors might cause failure.
    # Adding a tiny epsilon to the side length ensures validity.
    S_safe = S_opt * 1.0001
    
    inner = [[p[0], p[1], ir_opt] for p in points]
    center = [cx_opt, cy_opt]
    side = S_safe
    angle = phi_opt
    
    return inner, center, side, angle

