"""Packing 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import linprog

N = 11

_START = time.time()
_TIME_LIMIT = 140.0

def _hex_vertices(center, side, angle_deg):
    cx, cy = center
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _point_in_convex(px, py, verts):
    n = len(verts)
    area = 0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        area += x1 * y2 - x2 * y1
    ccw = area > 0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if ccw and cross > 1e-9:
            return False
        if not ccw and cross < -1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    return False

def _hexagons_overlap(hex1, hex2):
    v1 = _hex_vertices((hex1[0], hex1[1]), 1.0, hex1[2])
    v2 = _hex_vertices((hex2[0], hex2[1]), 1.0, hex2[2])
    for v in v1:
        if _point_in_convex(v[0], v[1], v2):
            return True
    for v in v2:
        if _point_in_convex(v[0], v[1], v1):
            return True
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _check_validity(inner, center, side, angle):
    outer_verts = _hex_vertices(center, side, angle)
    for i, (px, py, ang) in enumerate(inner):
        inner_verts = _hex_vertices((px, py), 1.0, ang)
        for v in inner_verts:
            if not _point_in_convex(v[0], v[1], outer_verts):
                return False
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if _hexagons_overlap(inner[i], inner[j]):
                return False
    return True

def _support_hex(angle_deg, direction_rad):
    """Support function of a unit regular hexagon with vertex angles at angle_deg
    in the given direction."""
    a = np.radians(angle_deg)
    diffs = a + np.arange(6) * np.pi / 3 - direction_rad
    return np.max(np.cos(diffs))

def _min_enclosing_hexagon_lp(positions, angles, outer_angle_deg):
    """Find min side s and center (cx,cy) of a regular hexagon with given outer_angle_deg
    that contains all unit hexagons at given positions/angles."""
    theta = np.radians(outer_angle_deg)
    normal_angles = theta + np.arange(6) * np.pi / 3
    
    n_inner = len(positions)
    A_rows = []
    b_vals = []
    
    for i in range(n_inner):
        px, py = positions[i]
        a_i = np.radians(angles[i])
        p = np.array([px, py])
        
        for k in range(6):
            n_k = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
            h_i = _support_hex(angles[i], normal_angles[k])
            
            # c . n_k + s * sqrt(3)/2 >= p . n_k + h_i
            A_row = np.array([n_k[0], n_k[1], np.sqrt(3)/2])
            b_val = p @ n_k + h_i
            A_rows.append(A_row)
            b_vals.append(b_val)
    
    A_mat = np.array(A_rows)
    b_vec = np.array(b_vals)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = -A_mat
    b_ub = -b_vec
    bounds = [(None, None), (None, None), (0.1, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.fun, (res.x[0], res.x[1])
    else:
        return float('inf'), (0.0, 0.0)

# EVOLVE-BLOCK-START
def construct():
    global _START, _TIME_LIMIT
    _START = time.time()
    
    sqrt3 = np.sqrt(3)
    
    # Generate multiple candidate cluster shapes
    candidates = []
    
    # Shape 1: 3-4-3 hexagonal cluster
    s1 = []
    for dx in [-1.5, 0.0, 1.5]:
        s1.append([dx, 0.0, 0.0])
    for dx in [-2.25, -0.75, 0.75, 2.25]:
        s1.append([dx, sqrt3, 0.0])
    for dx in [-1.5, 0.0, 1.5]:
        s1.append([dx, 2*sqrt3, 0.0])
    candidates.append(s1)
    
    # Shape 2: 4-3-4
    s2 = []
    for dx in [-2.25, -0.75, 0.75, 2.25]:
        s2.append([dx, 0.0, 0.0])
    for dx in [-1.5, 0.0, 1.5]:
        s2.append([dx, sqrt3, 0.0])
    for dx in [-2.25, -0.75, 0.75, 2.25]:
        s2.append([dx, 2*sqrt3, 0.0])
    candidates.append(s2)
    
    # Shape 3: More compact, overlapping rows
    s3 = []
    # Center hexagon
    s3.append([0.0, 0.0, 0.0])
    # 6 around it at distance sqrt(3)
    for k in range(6):
        ang = k * 60.0
        x = sqrt3 * np.cos(np.radians(ang))
        y = sqrt3 * np.sin(np.radians(ang))
        s3.append([x, y, 0.0])
    # 4 more to make 11
    # Place at second ring
    for k in range(4):
        ang = k * 90.0 + 45.0
        x = 2*sqrt3 * np.cos(np.radians(ang))
        y = 2*sqrt3 * np.sin(np.radians(ang))
        s3.append([x, y, 0.0])
    candidates.append(s3)
    
    best_side = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    for config in candidates:
        if time.time() - _START > _TIME_LIMIT - 15:
            break
        
        # Try different outer angles
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            if time.time() - _START > _TIME_LIMIT - 15:
                break
            
            positions = [(p[0], p[1]) for p in config]
            angles = [p[2] for p in config]
            
            s, center = _min_enclosing_hexagon_lp(positions, angles, outer_angle)
            
            if s < best_side:
                # Verify validity
                inner = [list(p) for p in config]
                if _check_validity(inner, center, s, outer_angle):
                    best_side = s
                    best_inner = inner
                    best_center = center
                    best_angle = outer_angle
    
    # Try to improve by adjusting rotations
    if best_inner is not None:
        for iteration in range(50):
            if time.time() - _START > _TIME_LIMIT - 5:
                break
            
            improved = False
            for i in range(len(best_inner)):
                for rot_delta in [30, -30, 60, -60]:
                    test_inner = [list(x) for x in best_inner]
                    test_inner[i][2] += rot_delta
                    
                    if _check_validity(test_inner, best_center, best_side, best_angle):
                        positions = [(p[0], p[1]) for p in test_inner]
                        angles = [p[2] for p in test_inner]
                        s_new, c_new = _min_enclosing_hexagon_lp(positions, angles, best_angle)
                        if s_new < best_side - 1e-6:
                            best_inner = test_inner
                            best_side = s_new
                            best_center = c_new
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    else:
        return candidates[0], (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
