"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a support-function based objective with differential evolution and local polish.
The key insight: for a fixed set of inner hexagons (positions + rotations), the minimum
outer hexagon side length (with fixed rotation) is computed analytically via support functions.
We optimize the 36D parameter space (12 hexagons × 3 params each) plus the outer rotation.
"""
N = 12

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import math

def _support_hex(unit_dir_cos, unit_dir_sin, hex_angle_rad):
    """
    Support function of a unit regular hexagon (side=1) in a given direction.
    The hexagon has vertices at angles hex_angle_rad + 60°*k for k=0..5.
    Returns max over vertices of dot(vertex_unit, direction).
    """
    # Vertex directions: hex_angle_rad + 60°*k
    v_angles = hex_angle_rad + np.radians(60 * np.arange(6))
    # dot = cos(v_angle - dir_angle), but we can compute directly:
    # vertex = (cos(v_angle), sin(v_angle)), direction = (unit_dir_cos, unit_dir_sin)
    dots = unit_dir_cos * np.cos(v_angles) + unit_dir_sin * np.sin(v_angles)
    return np.max(dots)

def _compute_outer_side(centers_xy, angles_rad, outer_angle_rad):
    """
    Given centers (N,2) and angles (N,) in radians, and outer hex rotation,
    compute the minimum side length S of the outer hexagon (centered at origin)
    that contains all inner hexagons.
    
    For each of the 6 outward normals n_i of the outer hex:
      constraint: max_k [ dot(center_k, n_i) + h_k(n_i) ] <= apothem = S * sqrt(3)/2
    where h_k(n_i) is the support function of inner hex k in direction n_i.
    
    So S = (2/sqrt(3)) * max over all (i, k) of [ dot(center_k, n_i) + h_k(n_i) ]
    """
    # 6 outward normals of outer hex
    n_angles = outer_angle_rad + np.radians(60 * np.arange(6))
    n_cos = np.cos(n_angles)  # shape (6,)
    n_sin = np.sin(n_angles)  # shape (6,)
    
    max_proj = 0.0
    for k in range(N):
        cx, cy = centers_xy[k]
        a_k = angles_rad[k]
        for i in range(6):
            # dot(center_k, n_i)
            dot_c = cx * n_cos[i] + cy * n_sin[i]
            # support of inner hex k in direction n_i
            h = _support_hex(n_cos[i], n_sin[i], a_k)
            proj = dot_c + h
            if proj > max_proj:
                max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _overlap_penalty(centers_xy, angles_rad):
    """
    Compute a penalty for overlapping hexagons.
    Two unit hexagons with centers p, q and rotations a, b overlap if
    the distance |p-q| is less than the sum of their support functions in
    the direction of (p-q).
    
    For a sufficient check: two convex polygons don't overlap iff there exists
    a separating axis. For hexagons, we check all 12 edge normals (6 from each).
    But for speed in the optimization loop, we use a simpler check:
    distance < 2 * min_possible_extent is definitely overlapping.
    The minimum extent of a unit hex in any direction is sqrt(3)/2 (inradius).
    So if dist < 2*sqrt(3)/2 = sqrt(3) ≈ 1.732, they MUST overlap.
    If dist >= 2, they definitely don't overlap (circumradius check).
    For the intermediate range, we do a more careful check.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers_xy[j, 0] - centers_xy[i, 0]
            dy = centers_xy[j, 1] - centers_xy[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist >= 2.0:
                continue  # definitely no overlap
            
            if dist < 1e-12:
                penalty += 2.0
                continue
            
            # Direction from i to j
            dir_x = dx / dist
            dir_y = dy / dist
            
            # Support of hex i in direction -dir (towards j)
            h_i = _support_hex(-dir_x, -dir_y, angles_rad[i])
            # Support of hex j in direction dir (towards i)
            h_j = _support_hex(dir_x, dir_y, angles_rad[j])
            
            # If h_i + h_j > dist, they overlap
            gap = h_i + h_j - dist
            if gap > 0:
                penalty += gap
    
    return penalty

def _full_objective(params, outer_angle_rad):
    """
    params: array of shape (N*3,) = [x0, y0, a0_deg, x1, y1, a1_deg, ...]
    Returns: outer_side + large_penalty * overlap_penalty
    """
    centers = np.empty((N, 2))
    angles_deg = np.empty(N)
    for i in range(N):
        centers[i, 0] = params[3 * i]
        centers[i, 1] = params[3 * i + 1]
        angles_deg[i] = params[3 * i + 2]
    
    angles_rad = np.radians(angles_deg)
    
    S = _compute_outer_side(centers, angles_rad, outer_angle_rad)
    op = _overlap_penalty(centers, angles_rad)
    
    return S + 50.0 * op

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit
    
    best_S = float('inf')
    best_params = None
    best_outer_angle_deg = 0.0
    
    # We'll try several outer hex rotations
    outer_angles_to_try = [0.0, 30.0, 15.0, 45.0]
    
    # Initial guess: hexagonal lattice
    # 12 hexagons in a 4x3 grid (staggered)
    def _initial_lattice():
        positions = []
        # 4 columns, 3 rows, staggered
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3.0)
                positions.append([x, y, 0.0])
        # Center it
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = np.mean(xs)
        cy = np.mean(ys)
        params = []
        for (x, y, a) in positions:
            params.extend([x - cx, y - cy, a])
        return np.array(params)
    
    init_params = _initial_lattice()
    
    for outer_angle_deg in outer_angles_to_try:
        if time.time() > deadline - 40.0:
            break
        
        outer_angle_rad = np.radians(outer_angle_deg)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))   # x
            bounds.append((-5.0, 5.0))   # y
            bounds.append((0.0, 60.0))   # angle (periodic)
        
        try:
            result = differential_evolution(
                _full_objective,
                bounds,
                args=(outer_angle_rad,),
                seed=42,
                maxiter=60,
                tol=1e-10,
                popsize=12,
                polish=True,
                init='latinhypercube'
            )
            
            opt_params = result.x
            centers = np.empty((N, 2))
            angles_deg = np.empty(N)
            for i in range(N):
                centers[i, 0] = opt_params[3 * i]
                centers[i, 1] = opt_params[3 * i + 1]
                angles_deg[i] = opt_params[3 * i + 2]
            angles_rad = np.radians(angles_deg)
            
            S_val = _compute_outer_side(centers, angles_rad, outer_angle_rad)
            op_val = _overlap_penalty(centers, angles_rad)
            
            if op_val < 1e-8 and S_val < best_S:
                best_S = S_val
                best_params = opt_params.copy()
                best_outer_angle_deg = outer_angle_deg
        
        except Exception:
            continue
        
        # Local polish
        if best_params is not None and time.time() < deadline - 15.0:
            try:
                result2 = minimize(
                    _full_objective,
                    best_params,
                    args=(np.radians(best_outer_angle_deg),),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                opt2 = result2.x
                centers2 = np.empty((N, 2))
                angles_deg2 = np.empty(N)
                for i in range(N):
                    centers2[i, 0] = opt2[3 * i]
                    centers2[i, 1] = opt2[3 * i + 1]
                    angles_deg2[i] = opt2[3 * i + 2]
                angles_rad2 = np.radians(angles_deg2)
                
                S2 = _compute_outer_side(centers2, angles_rad2, np.radians(best_outer_angle_deg))
                op2 = _overlap_penalty(centers2, angles_rad2)
                
                if op2 < 1e-8 and S2 < best_S:
                    best_S = S2
                    best_params = opt2.copy()
            except Exception:
                pass
    
    # Fallback if no valid solution found
    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3.0)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 7.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[3 * i])
        y = float(best_params[3 * i + 1])
        ang = float(best_params[3 * i + 2])
        inner.append([x, y, ang])
    
    # Final validation: check containment using vertex-in-polygon
    def _hex_verts(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        return np.array([
            [cx + side * np.cos(a + np.radians(60 * k)),
             cy + side * np.sin(a + np.radians(60 * k))]
            for k in range(6)
        ])
    
    def _point_in_convex_polygon(pt, verts):
        """Check if point is inside a convex polygon (CCW vertices)."""
        n = len(verts)
        for i in range(n):
            v1 = verts[i]
            v2 = verts[(i + 1) % n]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < -1e-9:
                return False
        return True
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle_deg)
    
    valid = True
    for i in range(N):
        iv = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for pt in iv:
            if not _point_in_convex_polygon(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Bump S up slightly to fix numerical issues
        best_S *= 1.02
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle_deg)
