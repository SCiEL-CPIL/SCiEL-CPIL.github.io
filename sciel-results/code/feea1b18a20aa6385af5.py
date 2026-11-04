import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    """
    Find the optimal rotation of the outer hexagon to minimize side length.
    """
    best_L = float('inf')
    best_rot = init_rot
    
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    # Local refinement
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    return best_L, best_rot

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

def smooth_overlap_penalty(params, beta=20.0):
    """
    Compute a smooth penalty for overlaps using log-sum-exp.
    Returns a scalar penalty that is 0 when no overlaps exist.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            # d is negative when overlapping
            penetrations.append(max(0.0, -d))
    
    if not penetrations:
        return 0.0
    
    penetrations = np.array(penetrations)
    max_pen = np.max(penetrations)
    
    if max_pen < 1e-12:
        return 0.0
    
    # Log-sum-exp for smooth max
    shifted = beta * (penetrations - max_pen)
    shifted = np.clip(shifted, -50, 50)
    log_sum = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    
    return log_sum

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    """Create a hex grid of unit hexagons with optional gap."""
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
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

def make_compact_cluster(gap=0.0):
    """
    Create a compact cluster of 12 hexagons arranged in a tighter pattern.
    Uses a 3-4-3-2 arrangement or similar to minimize bounding hexagon.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    
    # Row 0: 3 hexagons
    for c in range(3):
        pts.append([c * dx, 0, 0.0])
    
    # Row 1: 4 hexagons (offset)
    for c in range(4):
        pts.append([c * dx + dx/2, dy, 0.0])
    
    # Row 2: 3 hexagons
    for c in range(3):
        pts.append([c * dx, 2*dy, 0.0])
    
    # Row 3: 2 hexagons (offset)
    for c in range(2):
        pts.append([c * dx + dx/2, 3*dy, 0.0])
    
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_diamond(gap=0.0):
    """
    Diamond-shaped arrangement: 1-2-3-4-3-2-1 pattern (but we only need 12).
    Use 2-3-4-3 = 12.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    
    # Row 0: 2 hexagons
    for c in range(2):
        pts.append([c * dx + dx/2, 0, 0.0])
    
    # Row 1: 3 hexagons
    for c in range(3):
        pts.append([c * dx, dy, 0.0])
    
    # Row 2: 4 hexagons (offset)
    for c in range(4):
        pts.append([c * dx + dx/2, 2*dy, 0.0])
    
    # Row 3: 3 hexagons
    for c in range(3):
        pts.append([c * dx, 3*dy, 0.0])
    
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def is_valid(params, tol=1e-6):
    """Check if configuration has no overlaps."""
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def compute_penetrations(params):
    """Compute all pairwise penetration depths (negative = overlapping)."""
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            penetrations.append(d)
    return np.array(penetrations)

def residuals_least_squares(params, beta=20.0):
    """
    Smooth residuals for least_squares using soft-min for overlap constraints.
    Uses log-sum-exp style smoothing for the max penetration.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            penetrations.append(max(0.0, -d))
    
    if not penetrations:
        return np.zeros(1)
    
    penetrations = np.array(penetrations)
    max_pen = np.max(penetrations)
    
    if max_pen < 1e-12:
        return np.zeros(1)
    
    # Return a single residual based on smooth max
    shifted = beta * (penetrations - max_pen)
    shifted = np.clip(shifted, -50, 50)
    log_sum = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    
    return np.array([log_sum])

def objective_combined(params, penalty_weight=50.0, beta=20.0):
    """
    Combined objective: outer side length + smooth penalty for overlaps.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    # Compute smooth overlap penalty
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            penetrations.append(max(0.0, -d))
    
    penetrations = np.array(penetrations)
    max_pen = np.max(penetrations)
    
    if max_pen < 1e-12:
        penalty = 0.0
    else:
        shifted = beta * (penetrations - max_pen)
        shifted = np.clip(shifted, -50, 50)
        penalty = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    
    # Compute outer side length
    all_points = np.vstack(vertices_list)
    L = outer_hex_side(all_points, outer_rot)
    
    return L + penalty_weight * penalty

def sdf_distance_to_outer(point, outer_rot_deg):
    """
    Compute signed distance from point to boundary of unit outer hexagon
    centered at origin with given rotation. Positive = outside.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Distance from origin to each side along normal
    # For a regular hexagon with side length s, the apothem is s * sqrt(3)/2
    # We want the signed distance: positive if outside
    # Project point onto each normal, compare to apothem
    proj = point @ normals.T
    # The hexagon boundary is at distance apothem = s * sqrt(3)/2 from center
    # But we're computing for the enclosing hexagon, so we need to find
    # which side is closest
    # Actually, for a regular hexagon, the distance from center to side (apothem) 
    # is the same for all sides. The point is inside if proj < apothem for all sides.
    # Signed distance = max(0, max(proj) - apothem)
    # But we need the apothem of the outer hexagon.
    # This function is used for computing residuals, so let's just return
    # the max projection minus some reference.
    
    # Actually, let's compute the side length needed to enclose this point
    # and use that as a measure.
    # Simpler: just use the projection values as indicators
    max_proj = np.max(proj)
    return max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_compact_cluster(gap=gap))
    configs.append(make_diamond(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time_left() < 15:
            break
        
        for orot in outer_rots:
            if time_left() < 15:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                xys_rots = init_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Use least_squares to optimize
            if time_left() < 20:
                continue
            
            try:
                def residuals(params):
                    return residuals_least_squares(params)
                
                res_ls = least_squares(
                    residuals,
                    init_params,
                    method='trf',
                    max_nfev=200,
                    ftol=1e-8,
                    gtol=1e-8
                )
                
                if is_valid(res_ls.x):
                    xys_rots = res_ls.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_ls.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_ls.x.copy()
                        
            except Exception:
                pass
            
            # Nelder-Mead polish
            if time_left() < 25:
                continue
            
            try:
                polish_start = best_params if best_params is not None else init_params
                
                res_nm = minimize(
                    objective_combined,
                    polish_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 120,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                if is_valid(res_nm.x):
                    xys_rots = res_nm.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_nm.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
                        
            except Exception:
                pass
    
    # Optimize outer rotation independently
    if best_params is not None and time_left() > 25:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Final high-precision polish
    if best_params is not None and time_left() > 15:
        try:
            res_final = minimize(
                objective_combined,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_final.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # One more outer rotation optimization
    if best_params is not None and time_left() > 5:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
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
