import numpy as np
import time
import math
import random
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
    # Normals of the sides of the hexagon
    # A regular hexagon with side s has apothem a = s * sqrt(3) / 2
    # The distance from center to side is a.
    # If we project points onto the normals, the max projection must be <= a.
    # So a = max_proj.
    # s = 2 * a / sqrt(3).
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
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        cx = xys_rots[:, 0]
        cy = xys_rots[:, 1]
        
        # 1. Smooth Overlap Penalty using Inradius Circles
        # Inradius r = sqrt(3)/2. Min dist = 2*r = sqrt(3).
        # Penalty for pair i,j: max(0, sqrt(3) - dist_ij)^2
        cx_diff = cx[:, None] - cx[None, :]
        cy_diff = cy[:, None] - cy[None, :]
        dist_sq = cx_diff**2 + cy_diff**2
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_sq_pairs = dist_sq[i_idx, j_idx]
        d_pairs = np.sqrt(d_sq_pairs)
        violations = np.maximum(0.0, np.sqrt(3.0) - d_pairs)
        penalty = np.sum(violations**2)
        
        # 2. Enclosing Hexagon Side Length with smooth max
        vertices_list = get_hex_vertices_batch(xys_rots)
        all_points = np.vstack(vertices_list)
        
        angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        proj = all_points @ normals.T
        abs_proj = np.abs(proj)
        
        # Use log-sum-exp for smooth max with adaptive temperature
        tau = 0.05
        shifted = abs_proj / tau
        max_shifted = np.max(shifted)
        smooth_max = tau * (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))))
        
        L = (2.0 / np.sqrt(3.0)) * smooth_max
        
        # Weight the penalty more heavily to ensure feasibility
        return L + 50.0 * penalty

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

    def make_ring12(gap=0.0):
        """12 hexagons in a ring around center, rotated to face out."""
        pts = []
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_ring12_rotated(gap=0.0):
        """12 hexagons in a ring, rotated by 30 degrees relative to radial."""
        pts = make_ring12(gap)
        pts[:, 2] += 30.0
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
    configs.append(make_ring12(gap=gap))
    configs.append(make_ring12_rotated(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, -15.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 15:
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
                # Use Nelder-Mead with adaptive options - more iterations for better convergence
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1500,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
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
            # Use multiple restarts from the best solution with small perturbations
            for restart in range(3):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                if restart == 0:
                    start_params = best_params.copy()
                else:
                    start_params = best_params + np.random.normal(0, 0.01, size=best_params.shape)
                
                res = minimize(
                    objective,
                    start_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
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
            
    # Additional Polish: Optimize outer rotation for the fixed inner configuration
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        xys_rots = best_params[:3*N].reshape(N, 3)
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        def obj_rot(rot):
            return outer_hex_side(np.vstack(vertices_list), rot)
        
        try:
            # Grid search over rotations first
            best_rot = best_params[3*N]
            best_rot_L = obj_rot(best_rot)
            for rot in np.linspace(0, 60, 121):
                L_val = obj_rot(rot)
                if L_val < best_rot_L:
                    best_rot_L = L_val
                    best_rot = rot
            
            # Then refine with Nelder-Mead
            res_rot = minimize(
                obj_rot,
                best_rot,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            new_rot = res_rot.x[0]
            new_L = obj_rot(new_rot)
            
            if new_L < best_L:
                best_L = new_L
                best_params[3*N] = new_rot
        except Exception:
            pass

    # Fallback: Use the best valid initial config or a simple valid grid
    if best_params is None:
        # Create a simple valid ring configuration
        inner = []
        R = 2.5
        for k in range(N):
            theta = 2 * np.pi * k / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            inner.append([float(x), float(y), float(angle_deg)])
        # Calculate side for this fallback
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), float(L_val), 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle

if __name__ == "__main__":
    construct()
