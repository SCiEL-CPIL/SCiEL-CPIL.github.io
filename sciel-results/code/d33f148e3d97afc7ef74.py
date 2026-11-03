import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    N_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    # angles shape: (N, 6)
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)  # (N, 6)
    sin_a = np.sin(angles)  # (N, 6)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    # Return shape (N, 6, 2)
    return np.stack((x, y), axis=-1)

def hex_center_dist_valid(c1, c2, r1, r2):
    """
    For two regular hexagons of side 1, check if centers are far enough apart.
    The minimum center distance for non-overlapping unit hexagons depends on relative rotation.
    For same orientation: min dist = sqrt(3)
    We use a conservative check: if center dist < sqrt(3) - epsilon, they might overlap.
    For a more accurate check, we'd need full SAT, but this is a fast pre-filter.
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    d = np.sqrt(dx*dx + dy*dy)
    # Conservative: if d < sqrt(3), they might overlap (for aligned hexes)
    # For rotated hexes, the min distance can be smaller, but let's use SAT for accuracy
    return d

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def sat_overlap_batch(vertices_list):
    """
    Vectorized SAT overlap check for all pairs.
    Returns (has_overlap, penalty) where penalty is sum of squared penetration depths.
    """
    N_pts = len(vertices_list)
    # Precompute axes for all hexagons
    # For each hexagon i, compute its 6 edge normals
    axes_all = []
    for v in vertices_list:
        hex_axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                hex_axes.append(n)
        axes_all.append(hex_axes)
    
    has_overlap = False
    penalty = 0.0
    
    for i in range(N_pts):
        for j in range(i+1, N_pts):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Combine axes from both polygons
            all_axes = axes_all[i] + axes_all[j]
            
            min_overlap = float('inf')
            separated = False
            for n in all_axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                
                overlap = min(max1, max2) - max(min1, min2)
                min_overlap = min(min_overlap, overlap)
            
            if not separated:
                if min_overlap > 1e-9:
                    has_overlap = True
                    penalty += min_overlap ** 2
    
    return has_overlap, penalty

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        # Overlap penalty using batch SAT
        has_overlap, penalty = sat_overlap_batch(vertices_list)
        
        L = outer_hex_side_batch(vertices_list, outer_rot)
        
        return L + 100.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5 # Approximate gap in y
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        """1 center + 6 ring + 5 outer with gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations with a small gap to ensure validity
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            xys_rots_init = init_params[:3*N].reshape(N, 3)
            outer_rot_init = init_params[3*N]
            vertices_list_init = get_hex_vertices_batch(xys_rots_init)
            
            has_overlap, _ = sat_overlap_batch(vertices_list_init)
            
            if not has_overlap:
                L_val = outer_hex_side_batch(vertices_list_init, outer_rot_init)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            try:
                # Use Nelder-Mead with adaptive options
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                # Evaluate the result
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices_list = get_hex_vertices_batch(xys_rots)
                
                # Check for overlaps
                has_overlap, _ = sat_overlap_batch(vertices_list)
                
                if not has_overlap:
                    L_val = outer_hex_side_batch(vertices_list, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # If we have a good result, do a final refinement with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            
            has_overlap, _ = sat_overlap_batch(vertices_list)
            
            if not has_overlap:
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Fallback: Use the best valid initial config or a simple valid grid
    if best_params is None:
        # Create a simple valid grid with large gaps
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        # Calculate side for this fallback
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
