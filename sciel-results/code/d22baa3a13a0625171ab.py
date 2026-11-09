"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a dual ascent / hard-core potential approach with global exploration.
"""
N = 12

import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(cx, cy, angle_deg, side, dx, dy):
    """Support function of a regular hexagon in direction (dx, dy)."""
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(cx, cy, ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            inside = True
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j+1)%6]
                edge = v2 - v1
                rel = pt - v1
                cross = edge[0]*rel[1] - edge[1]*rel[0]
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            # Check vertex-in-polygon
            for pt in vi:
                inside = True
                for k in range(6):
                    v1 = vj[k]
                    v2 = vj[(k+1)%6]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -1e-9:
                        inside = False
                        break
                if inside:
                    return False
            
            for pt in vj:
                inside = True
                for k in range(6):
                    v1 = vi[k]
                    v2 = vi[(k+1)%6]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -1e-9:
                        inside = False
                        break
                if inside:
                    return False
            
            # Check edge intersections
            for i_e in range(6):
                p1 = vi[i_e]
                p2 = vi[(i_e+1)%6]
                for j_e in range(6):
                    p3 = vj[j_e]
                    p4 = vj[(j_e+1)%6]
                    
                    def cross(o, a, b):
                        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                    
                    d1 = cross(p3, p4, p1)
                    d2 = cross(p3, p4, p2)
                    d3 = cross(p1, p2, p3)
                    d4 = cross(p1, p2, p4)
                    
                    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                        return False
    
    return True

def _overlap_depth_hex(cx1, cy1, ang1, cx2, cy2, ang2):
    """
    Estimate overlap depth between two unit hexagons using support functions.
    Returns 0 if no overlap, positive value if overlapping.
    """
    # For two convex polygons, the minimum translation distance (MTD) can be
    # approximated by checking all edge normals of both polygons.
    # If all projections are separated, no overlap.
    # The overlap depth is the minimum over all separating axes of the 
    # negative of the gap (if gap < 0, they overlap).
    
    # Get edge normals for hex1
    a1 = np.radians(ang1)
    normals1 = np.array([
        [np.cos(a1 + np.radians(60 * i + 30)),
         np.sin(a1 + np.radians(60 * i + 30))]
        for i in range(6)
    ])
    
    a2 = np.radians(ang2)
    normals2 = np.array([
        [np.cos(a2 + np.radians(60 * i + 30)),
         np.sin(a2 + np.radians(60 * i + 30))]
        for i in range(6)
    ])
    
    all_normals = np.vstack([normals1, normals2])
    
    c1 = np.array([cx1, cy1])
    c2 = np.array([cx2, cy2])
    d = c2 - c1
    
    min_gap = np.inf
    for n in all_normals:
        # Project both hexagons onto this normal
        s1 = _support_hex(cx1, cy1, ang1, 1.0, n[0], n[1])
        s2 = _support_hex(cx2, cy2, ang2, 1.0, n[0], n[1])
        
        # The extent of hex1 in direction n from center
        # The extent of hex2 in direction -n from center
        proj1 = s1  # max projection of hex1 in direction n
        proj2 = -s2  # max projection of hex2 in direction -n (i.e., -n)
        
        # Gap along this axis
        # If proj1 + proj2 < |d| in that direction, they are separated
        dot_d = d @ n
        gap = proj1 + proj2 - abs(dot_d)
        
        if gap < min_gap:
            min_gap = gap
    
    if min_gap > 0:
        return min_gap  # Overlap depth
    else:
        return 0.0  # No overlap

def _hard_core_potential(centers, angles, S, outer_angle_deg=0.0, beta=100.0):
    """
    Compute a hard-core potential: sum of overlap penalties between inner hexagons
    and containment violations with the outer hexagon.
    """
    penalty = 0.0
    
    # Containment penalty
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    for k in range(N):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(cx, cy, ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            apothem = S * np.sqrt(3.0) / 2.0
            if needed > apothem:
                penalty += beta * (needed - apothem)**2
    
    # Overlap penalty between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            depth = _overlap_depth_hex(
                centers[i, 0], centers[i, 1], angles[i],
                centers[j, 0], centers[j, 1], angles[j]
            )
            if depth > 0:
                penalty += beta * depth**2
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_inner = None
    best_S = np.inf
    best_angles = None
    
    # Strategy 1: Try aligned lattice configurations with optimization
    # Initial lattice
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    def _make_lattice(scale, offset_x=0.0, offset_y=0.0):
        """Create a 4x3 lattice of hexagons."""
        positions = []
        for i in range(4):
            for j in range(3):
                pos = scale * (i * u + j * v) + np.array([offset_x, offset_y])
                positions.append(pos)
        # Center the configuration
        mean = np.mean(positions, axis=0)
        positions = [p - mean for p in positions]
        return np.array(positions)
    
    # Try multiple scales
    for trial_scale in [0.98, 1.0, 1.02, 1.05, 1.1]:
        if time.time() > deadline - 30.0:
            break
        
        centers = _make_lattice(trial_scale)
        angles = np.zeros(N)
        
        S = _min_enclosing_hex_side(centers, angles, 0.0)
        
        # Check validity
        if _check_validity(centers, angles, S, 0.0):
            if S < best_S:
                best_S = S
                best_inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
                best_angles = angles.copy()
        
        # Local refinement using Nelder-Mead
        x0 = np.concatenate([centers.flatten()])
        
        def objective(params):
            c = params[:2*N].reshape(N, 2)
            S_val = _min_enclosing_hex_side(c, np.zeros(N), 0.0)
            pen = _hard_core_potential(c, np.zeros(N), S_val, 0.0, beta=50.0)
            return S_val + pen
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
            c_opt = res.x[:2*N].reshape(N, 2)
            S_opt = _min_enclosing_hex_side(c_opt, np.zeros(N), 0.0)
            
            if _check_validity(c_opt, np.zeros(N), S_opt, 0.0):
                if S_opt < best_S:
                    best_S = S_opt
                    best_inner = [[float(cx), float(cy), 0.0] for cx, cy in c_opt]
                    best_angles = np.zeros(N)
        except Exception:
            pass
    
    # Strategy 2: Differential evolution for global exploration
    if time.time() < deadline - 60.0:
        # Define bounds for positions
        # A reasonable bounding box for the centers
        bound = 5.0
        bounds = [(-bound, bound)] * (2*N)
        
        def de_objective(params):
            c = params[:2*N].reshape(N, 2)
            S_val = _min_enclosing_hex_side(c, np.zeros(N), 0.0)
            pen = _hard_core_potential(c, np.zeros(N), S_val, 0.0, beta=20.0)
            return S_val + pen
        
        try:
            de_res = differential_evolution(
                de_objective, bounds,
                maxiter=100, popsize=15,
                tol=1e-4, seed=42,
                polish=True,
                updating='deferred'
            )
            
            if de_res.success or de_res.fun < best_S:
                c_de = de_res.x[:2*N].reshape(N, 2)
                S_de = _min_enclosing_hex_side(c_de, np.zeros(N), 0.0)
                
                if _check_validity(c_de, np.zeros(N), S_de, 0.0):
                    if S_de < best_S:
                        best_S = S_de
                        best_inner = [[float(cx), float(cy), 0.0] for cx, cy in c_de]
                        best_angles = np.zeros(N)
        except Exception:
            pass
    
    # Strategy 3: Ring-based initialization
    if time.time() < deadline - 40.0:
        # Place 6 hexagons in a ring and 6 in the center
        # Ring radius
        ring_r = 2.5
        center_r = 0.8
        
        centers_ring = []
        for i in range(6):
            ang = np.radians(60 * i)
            centers_ring.append([ring_r * np.cos(ang), ring_r * np.sin(ang)])
        
        centers_center = []
        for i in range(6):
            ang = np.radians(60 * i + 30)
            centers_center.append([center_r * np.cos(ang), center_r * np.sin(ang)])
        
        centers = np.array(centers_ring + centers_center)
        angles = np.zeros(N)
        
        S = _min_enclosing_hex_side(centers, angles, 0.0)
        
        if _check_validity(centers, angles, S, 0.0):
            if S < best_S:
                best_S = S
                best_inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
                best_angles = angles.copy()
        
        # Refine
        x0 = centers.flatten()
        
        def objective_ring(params):
            c = params[:2*N].reshape(N, 2)
            S_val = _min_enclosing_hex_side(c, np.zeros(N), 0.0)
            pen = _hard_core_potential(c, np.zeros(N), S_val, 0.0, beta=50.0)
            return S_val + pen
        
        try:
            res = minimize(objective_ring, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
            c_opt = res.x[:2*N].reshape(N, 2)
            S_opt = _min_enclosing_hex_side(c_opt, np.zeros(N), 0.0)
            
            if _check_validity(c_opt, np.zeros(N), S_opt, 0.0):
                if S_opt < best_S:
                    best_S = S_opt
                    best_inner = [[float(cx), float(cy), 0.0] for cx, cy in c_opt]
                    best_angles = np.zeros(N)
        except Exception:
            pass
    
    # Fallback: return a valid configuration if we haven't found one
    if best_inner is None:
        # Use the simplest lattice
        centers = _make_lattice(1.02)
        angles = np.zeros(N)
        S = _min_enclosing_hex_side(centers, angles, 0.0)
        best_inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
        best_S = S
    
    return best_inner, (0.0, 0.0), float(best_S), 0.0
