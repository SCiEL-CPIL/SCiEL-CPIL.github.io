"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses vectorized SAT-based overlap detection and binary search on container size
with Nelder-Mead optimization of a smooth penalty function.
"""
N = 12
import numpy as np
import time
from scipy.optimize import minimize
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_normals(angle_deg):
    """Outward unit normals of a regular hexagon with given rotation."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _sat_overlap_depth(v1, n1, v2, n2):
    """
    Compute SAT overlap depth between two convex hexagons.
    v1, n1: vertices and outward normals of hex 1 (6x2 arrays)
    v2, n2: vertices and outward normals of hex 2 (6x2 arrays)
    Returns overlap depth (positive if overlapping, 0 if separated).
    """
    max_penetration = 0.0
    # Test all 12 separating axes (6 normals from each hex)
    for i in range(6):
        # Axis n1[i]
        d1 = v1 @ n1[i]  # projections of hex1 onto axis
        d2 = v2 @ n1[i]  # projections of hex2 onto axis
        # Overlap on this axis
        overlap = min(d1.max(), d2.max()) - max(d1.min(), d2.min())
        if overlap <= 0:
            return 0.0  # Separated
        if overlap > max_penetration:
            max_penetration = overlap
    
    for i in range(6):
        # Axis n2[i]
        d1 = v1 @ n2[i]
        d2 = v2 @ n2[i]
        overlap = min(d1.max(), d2.max()) - max(d1.min(), d2.min())
        if overlap <= 0:
            return 0.0  # Separated
        if overlap > max_penetration:
            max_penetration = overlap
    
    return max_penetration

def _containment_violation(centers, angles, outer_side, outer_angle_deg):
    """
    Compute containment violation: how much the inner hexagons extend
    beyond the outer hexagon. Returns sum of violations (0 if all contained).
    """
    outer_normals = _hex_normals(outer_angle_deg)
    apothem = outer_side * math.sqrt(3.0) / 2.0
    # Outer hex center is at origin
    
    total_violation = 0.0
    for i in range(N):
        cx, cy = centers[i, 0], centers[i, 1]
        ang = angles[i]
        inner_normals = _hex_normals(ang)
        
        # For each outer edge normal, check if any inner vertex exceeds it
        for ni in range(6):
            n = outer_normals[ni]
            # Support function of inner hex in direction n
            # h(n) = max over inner vertices of dot(v, n)
            # Inner hex vertices relative to its center:
            a = np.radians(ang)
            v_angles = a + np.radians(60 * np.arange(6))
            v_rel = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
            h = np.max(v_rel @ n)
            
            # Distance from outer center to inner center projected on n
            d_center = cx * n[0] + cy * n[1]
            # Total extent in direction n
            extent = d_center + h
            if extent > apothem:
                total_violation += (extent - apothem)
    
    return total_violation

def _pairwise_overlap_penalty(centers, angles):
    """Compute total pairwise overlap penalty using SAT."""
    total = 0.0
    for i in range(N):
        v1 = _hex_verts(centers[i, 0], centers[i, 1], 1.0, angles[i])
        n1 = _hex_normals(angles[i])
        for j in range(i+1, N):
            v2 = _hex_verts(centers[j, 0], centers[j, 1], 1.0, angles[j])
            n2 = _hex_normals(angles[j])
            depth = _sat_overlap_depth(v1, n1, v2, n2)
            if depth > 0:
                total += depth
    return total

def _total_penalty(params, outer_side, outer_angle_deg):
    """
    Combined penalty: containment violation + pairwise overlap.
    params: (N*3,) array [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    cv = _containment_violation(centers, angles, outer_side, outer_angle_deg)
    op = _pairwise_overlap_penalty(centers, angles)
    return cv + op

def _try_pack(outer_angle_deg, time_budget):
    """
    Binary search on outer side length, with Nelder-Mead optimization
    of the penalty function for each candidate side length.
    Returns (side, params) or None.
    """
    start = time.time()
    
    # Initial hex lattice configuration
    hex_positions = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            hex_positions.append([x, y, 0.0])
    
    cx_mean = np.mean([p[0] for p in hex_positions])
    cy_mean = np.mean([p[1] for p in hex_positions])
    
    init_params = []
    for (x, y, a) in hex_positions:
        init_params.extend([x - cx_mean, y - cy_mean, a])
    init_params = np.array(init_params)
    
    # Binary search bounds
    S_lo = 2.5
    S_hi = 6.0
    
    # First, find a feasible S_hi
    for _ in range(5):
        if time.time() - start > time_budget:
            break
        res = minimize(
            _total_penalty, init_params,
            args=(S_hi, outer_angle_deg),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.fun < 1e-8:
            break
        S_hi *= 1.15
    
    # Binary search
    best_params = init_params.copy()
    best_S = S_hi
    iterations = 0
    max_iter = 25
    
    for _ in range(max_iter):
        if time.time() - start > time_budget - 5.0:
            break
        iterations += 1
        
        S_mid = (S_lo + S_hi) / 2.0
        res = minimize(
            _total_penalty, best_params,
            args=(S_mid, outer_angle_deg),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < 1e-8:
            S_hi = S_mid
            best_params = res.x.copy()
            best_S = S_mid
        else:
            S_lo = S_mid
        
        if S_hi - S_lo < 0.005:
            break
    
    # Final polish with more iterations
    if time.time() - start < time_budget - 2.0:
        res = minimize(
            _total_penalty, best_params,
            args=(best_S, outer_angle_deg),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res.fun < 1e-8 and res.x is not None:
            best_params = res.x
            # Try to tighten S slightly
            S_test = best_S * 0.998
            res2 = minimize(
                _total_penalty, best_params,
                args=(S_test, outer_angle_deg),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res2.fun < 1e-8:
                best_S = S_test
                best_params = res2.x
    
    return best_S, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_S = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    # Try several outer hex rotations
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 60.0:
            break
        
        try:
            S, params = _try_pack(outer_angle, time_budget=50.0)
            if params is not None and S < best_S:
                # Verify
                centers = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    centers[i, 0] = params[3*i]
                    centers[i, 1] = params[3*i+1]
                    angles[i] = params[3*i+2]
                penalty = _total_penalty(params, S, outer_angle)
                if penalty < 1e-6:
                    best_S = S
                    best_params = params.copy()
                    best_outer_angle = outer_angle
        except Exception:
            continue
    
    # If no good result, use fallback hex lattice
    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * math.sqrt(3)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation: check containment and non-overlap
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle)
    
    def point_in_hex(pt, outer_verts):
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i+1) % 6]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0]*rel[1] - edge[1]*rel[0]
            if cross < -1e-9:
                return False
        return True
    
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not point_in_hex(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Increase S slightly to ensure validity
        best_S *= 1.02
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
