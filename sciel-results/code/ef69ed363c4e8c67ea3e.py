import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _all_vertices(inner_params):
    """Compute all vertices for all inner hexagons. Returns (n, 6, 2)."""
    n = len(inner_params)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
    return verts

def _compute_side_fixed_angle(verts, outer_angle_deg):
    """
    Fast computation of minimum outer hexagon side length for a FIXED outer angle.
    verts: (n, 6, 2) array of all inner hexagon vertices.
    outer_angle_deg: rotation of outer hexagon.
    Returns the minimum side length S.
    """
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    # Face normals of outer hexagon (perpendicular to edges)
    # For a regular hexagon with vertices at 0, 60, ..., the normals are at 30, 90, ...
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # Flatten vertices: (n*6, 2)
    flat = verts.reshape(-1, 2)
    
    # For each normal, compute max dot product over all points
    # dots: (6, n*6)
    dots = normals @ flat.T
    max_dots = np.max(dots, axis=1)  # (6,)
    
    # side = max(max_dots) / apothem, where apothem = sqrt(3)/2
    return np.max(max_dots) / (SQRT3 / 2.0)

def _sat_overlap(v1, v2, tol=1e-9):
    """
    Check if two convex hexagons overlap using Separating Axis Theorem.
    v1, v2: (6, 2) arrays of vertices.
    Returns True if they overlap, False otherwise.
    """
    polys = [v1, v2]
    # Check axes from both polygons
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            # Check for separation
            if proj1.max() + tol < proj2.min() or proj2.max() + tol < proj1.min():
                return False
    return True

def _overlap_penalty_fast(verts, penalty_weight=1000.0):
    """
    Compute penalty for overlapping inner hexagons using SAT.
    verts: (n, 6, 2)
    """
    n = len(verts)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            # Quick center distance check to avoid SAT calls when far apart
            c1 = verts[i].mean(axis=0)
            c2 = verts[j].mean(axis=0)
            dist = np.linalg.norm(c1 - c2)
            # Max distance between centers for two unit hexagons to touch is roughly 2*sqrt(3) ~ 3.46
            # If distance > 3.5, they definitely don't overlap
            if dist < 3.5:
                if _sat_overlap(verts[i], verts[j]):
                    # Penalty based on how deep the overlap is (approximated by center distance reduction)
                    # A simple constant penalty is often sufficient for optimization
                    penalty += penalty_weight
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective: outer side length + overlap penalty.
    params: flattened array of (x, y, angle) for n hexagons.
    """
    inner_params = params.reshape(n, 3)
    verts = _all_vertices(inner_params)
    side = _compute_side_fixed_angle(verts, outer_angle_deg)
    overlap_pen = _overlap_penalty_fast(verts)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=1e-9):
    """Check if configuration is valid: all inside outer, no overlaps."""
    n = len(inner_params)
    # Check bounds
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    verts = _all_vertices(inner_params)
    
    # Point in polygon check for all vertices
    for i in range(n):
        for v in verts[i]:
            if not _point_in_polygon(v, outer_v):
                return False
    
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            if _sat_overlap(verts[i], verts[j]):
                return False
    return True

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Calculate side length for seed
    seed_inner_arr = np.array(inner)
    verts = _all_vertices(seed_inner_arr)
    side = _compute_side_fixed_angle(verts, 30.0) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    safety_margin = 5.0
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    
    # Phase 1: Powell Optimization
    # Powell is robust for non-smooth functions (max + penalty)
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            res = minimize(_objective, best_inner.flatten(), args=(outer_angle, n),
                          method='Powell',
                          options={'maxiter': 200, 'maxfev': 5000, 'disp': False})
            
            if time.time() - start_time < DEADLINE - safety_margin:
                candidate = res.x.reshape(n, 3)
                cand_side = _compute_side_fixed_angle(_all_vertices(candidate), outer_angle)
                # Check validity
                if _is_valid(candidate, cand_side, outer_angle, min_sep=1e-9):
                    if cand_side < best_side:
                        best_inner = candidate
                        best_side = cand_side
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead for local refinement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            res2 = minimize(_objective, best_inner.flatten(), args=(outer_angle, n),
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if time.time() - start_time < DEADLINE - safety_margin:
                candidate = res2.x.reshape(n, 3)
                cand_side = _compute_side_fixed_angle(_all_vertices(candidate), outer_angle)
                if _is_valid(candidate, cand_side, outer_angle, min_sep=1e-9):
                    if cand_side < best_side:
                        best_inner = candidate
                        best_side = cand_side
        except Exception:
            pass
            
    # Phase 3: High precision polish if time remains
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            res3 = minimize(_objective, best_inner.flatten(), args=(outer_angle, n),
                           method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if time.time() - start_time < DEADLINE - safety_margin:
                candidate = res3.x.reshape(n, 3)
                cand_side = _compute_side_fixed_angle(_all_vertices(candidate), outer_angle)
                if _is_valid(candidate, cand_side, outer_angle, min_sep=1e-9):
                    if cand_side < best_side:
                        best_inner = candidate
                        best_side = cand_side
        except Exception:
            pass

    # Final validation
    final_side = _compute_side_fixed_angle(_all_vertices(best_inner), outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        best_inner = np.array(seed_inner)
        final_side = seed_side

    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
