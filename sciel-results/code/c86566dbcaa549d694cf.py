import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _check_disjoint_fast(inner_positions, inner_angles, tol=1e-10):
    """Check disjointness using minimum distance between convex polygons."""
    n = len(inner_positions)
    all_verts = []
    for i in range(n):
        px, py = inner_positions[i]
        ang = inner_angles[i]
        all_verts.append(_get_hex_vertices(px, py, ang))
    
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap_fast(all_verts[i], all_verts[j]):
                return False
    return True

def _hexagons_overlap_fast(verts1, verts2):
    """Check if two convex hexagons overlap using SAT (Separating Axis Theorem)."""
    # For two convex polygons, they overlap iff no separating axis exists
    # Try all edge normals from both polygons
    for verts in [verts1, verts2]:
        for k in range(6):
            a = verts[k]
            b = verts[(k + 1) % 6]
            # Edge normal
            ex = b[0] - a[0]
            ey = b[1] - a[1]
            nx = -ey
            ny = ex
            # Project both polygons onto this normal
            min1 = min(v[0] * nx + v[1] * ny for v in verts1)
            max1 = max(v[0] * nx + v[1] * ny for v in verts1)
            min2 = min(v[0] * nx + v[1] * ny for v in verts2)
            max2 = max(v[0] * nx + v[1] * ny for v in verts2)
            # If there's a gap, they don't overlap
            if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                return False
    return True

def _min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find minimum side length of regular hexagon with given rotation
    that contains all vertices. Uses LP.
    Returns (cx, cy, side) or None.
    """
    # A regular hexagon with center c and side s has inradius r = s * sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes:
    # For each of 6 directions at 30+60k degrees: |n_k . (x - c)| <= r
    
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            # n . (c - v) <= r  =>  nx*cx + ny*cy - r <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            # -n . (c - v) <= r  =>  -nx*cx - ny*cy - r <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
    
    # r >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, r = res.x
            side = r / (math.sqrt(3) / 2)
            return cx, cy, side
        return None
    except:
        return None

def _compute_all_vertices(positions, angles):
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        all_verts.extend(_get_hex_vertices(px, py, ang))
    return all_verts

def _evaluate_configuration(positions, angles):
    """Evaluate a configuration: check disjointness and find best outer hex."""
    if not _check_disjoint_fast(positions, angles):
        return None, None, None, float('inf')
    
    all_verts = _compute_all_vertices(positions, angles)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try outer angles from 0 to 30 (by symmetry)
    for outer_angle in np.arange(0, 30.01, 1.0):
        res = _min_enclosing_hex_side_lp(all_verts, outer_angle)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = outer_angle
    
    # Fine tune around best angle
    if best_angle > 0:
        for outer_angle in np.arange(max(0, best_angle - 1.0), 
                                     min(30, best_angle + 1.0) + 0.01, 0.05):
            res = _min_enclosing_hex_side_lp(all_verts, outer_angle)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = outer_angle
    
    return positions, angles, best_center, best_side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback: line arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    best_side = float('inf')
    best_result = None
    
    # Strategy 1: Symmetric hexagonal cluster
    # Center + 6 neighbors at distance d1 + 4 at distance d2 at specific angles
    # For unit hexagons, minimum center-to-center distance for non-overlap is 2*sqrt(3)/2 = sqrt(3)
    # when aligned, but can be less with rotation
    
    # Generate candidate configurations
    
    # Config 1: Perfect hexagonal lattice (touching)
    d1 = sqrt3  # distance to first ring
    d2 = 2 * sqrt3  # distance to second ring
    
    # First ring: 6 hexagons at 60-degree intervals
    # Second ring: we need 4 more, place them at specific positions
    
    # Let's try: center + 6 at radius d1 + 4 at radius d2
    # The 4 in second ring should be at angles 0, 120, 240, 300 or 60, 180, 300, 120
    
    configs = []
    
    # Config A: All aligned (angle=0)
    # Center at origin
    # 6 at radius d1, angles 0,60,120,180,240,300
    # 4 at radius d2, angles 0,120,240,300
    for second_ring_angles in [[0, 120, 240, 300], [60, 180, 300, 120]]:
        positions = [(0.0, 0.0)]
        angles = [0.0]
        
        for k in range(6):
            a = math.radians(60 * k)
            positions.append((d1 * math.cos(a), d1 * math.sin(a)))
            angles.append(0.0)
        
        for a_deg in second_ring_angles:
            a = math.radians(a_deg)
            positions.append((d2 * math.cos(a), d2 * math.sin(a)))
            angles.append(0.0)
        
        configs.append((positions, angles))
    
    # Config B: Rotated inner hexagons for tighter packing
    # Try rotating some hexagons by 30 degrees
    for second_ring_angles in [[0, 120, 240, 300], [60, 180, 300, 120]]:
        for rot_pattern in [[0.0] * 11, [0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0]]:
            positions = [(0.0, 0.0)]
            angles = [rot_pattern[0]]
            
            for k in range(6):
                a = math.radians(60 * k)
                positions.append((d1 * math.cos(a), d1 * math.sin(a)))
                angles.append(rot_pattern[1 + k])
            
            for i, a_deg in enumerate(second_ring_angles):
                a = math.radians(a_deg)
                positions.append((d2 * math.cos(a), d2 * math.sin(a)))
                angles.append(rot_pattern[7 + i])
            
            configs.append((positions, angles))
    
    # Config C: Slightly compressed lattice
    for scale in [0.98, 0.99, 1.0, 1.01]:
        d1s = d1 * scale
        d2s = d2 * scale
        for second_ring_angles in [[0, 120, 240, 300], [60, 180, 300, 120]]:
            positions = [(0.0, 0.0)]
            angles = [0.0]
            
            for k in range(6):
                a = math.radians(60 * k)
                positions.append((d1s * math.cos(a), d1s * math.sin(a)))
                angles.append(0.0)
            
            for a_deg in second_ring_angles:
                a = math.radians(a_deg)
                positions.append((d2s * math.cos(a), d2s * math.sin(a)))
                angles.append(0.0)
            
            configs.append((positions, angles))
    
    # Evaluate all configs
    for positions, angles in configs:
        if time.time() > _DEADLINE - 20.0:
            break
        
        pos, ang, center, side = _evaluate_configuration(positions, angles)
        if pos is not None and side < best_side:
            best_side = side
            best_result = (pos, ang, center)
    
    # Local optimization on best configuration
    if best_result is not None and time.time() < _DEADLINE - 30.0:
        positions0, angles0, _ = best_result
        
        # Flatten parameters
        x0 = []
        for px, py in positions0:
            x0.extend([px, py])
        x0.extend(angles0)
        
        def objective(params):
            n = len(positions0)
            pos = [(params[2*i], params[2*i+1]) for i in range(n)]
            ang = list(params[2*n:])
            
            if not _check_disjoint_fast(pos, ang):
                return 1e6
            
            all_verts = _compute_all_vertices(pos, ang)
            
            best = float('inf')
            for outer_angle in np.arange(0, 30.01, 2.0):
                res = _min_enclosing_hex_side_lp(all_verts, outer_angle)
                if res:
                    if res[2] < best:
                        best = res[2]
            return best
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.x is not None:
                new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
                new_ang = list(res.x[2*N:])
                
                if _check_disjoint_fast(new_pos, new_ang):
                    pos, ang, center, side = _evaluate_configuration(new_pos, new_ang)
                    if pos is not None and side < best_side:
                        best_side = side
                        best_result = (pos, ang, center)
        except:
            pass
    
    # Final fine-tune of outer angle
    if best_result is not None:
        positions0, angles0, _ = best_result
        all_verts = _compute_all_vertices(positions0, angles0)
        
        best_final_side = float('inf')
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
        
        for outer_angle in np.arange(0, 30.01, 0.01):
            if time.time() > _DEADLINE - 5.0:
                break
            res = _min_enclosing_hex_side_lp(all_verts, outer_angle)
            if res:
                cx, cy, side = res
                if side < best_final_side:
                    best_final_side = side
                    best_final_center = (cx, cy)
                    best_final_angle = outer_angle
        
        if best_final_side < best_side:
            best_side = best_final_side
            best_result = (positions0, angles0, best_final_center)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, center = best_result
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    return inner, center, best_side, best_final_angle if best_result[2] != center else 0.0
