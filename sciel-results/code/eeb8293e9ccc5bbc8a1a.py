import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_convex_polygon(p, polygon):
    px, py = p
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            if sign > 0:
                return False
            sign = -1
        elif cross > 1e-9:
            if sign < 0:
                return False
            sign = 1
    return True

def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    if abs(c1) < 1e-9 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-9 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-9 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-9 and _on_segment(p1, p2, p4):
        return True
    return False

def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def _hexagons_overlap(verts1, verts2):
    if _point_in_convex_polygon(verts1[0], verts2):
        return True
    if _point_in_convex_polygon(verts2[0], verts1):
        return True
    for k in range(6):
        a1 = verts1[k]
        b1 = verts1[(k + 1) % 6]
        for l in range(6):
            a2 = verts2[l]
            b2 = verts2[(l + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _min_outer_side_lp(vertices, outer_angle_deg):
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
    
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t * 2.0 / math.sqrt(3)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _find_best_outer_lp(vertices, time_budget=1.0):
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Search 0 to 60 degrees
    for outer_angle_deg in np.arange(0, 60.1, 5.0):
        if time.time() - start > time_budget:
            break
        res = _min_outer_side_lp(vertices, outer_angle_deg)
        if res:
            side = res[2]
            if side < best_side:
                best_side = side
                best_center = (res[0], res[1])
                best_angle = outer_angle_deg
                
    # Refine
    if best_side < float('inf') and time.time() - start < time_budget:
        for outer_angle_deg in np.arange(max(0, best_angle - 5.0), min(60, best_angle + 5.0) + 0.1, 0.5):
            if time.time() - start > time_budget:
                break
            res = _min_outer_side_lp(vertices, outer_angle_deg)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
                    best_center = (res[0], res[1])
                    best_angle = outer_angle_deg
                    
    return best_side, best_center, best_angle

def _generate_initial_positions():
    # Hexagonal lattice
    sqrt3 = math.sqrt(3)
    positions = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            positions.append((x, y))
            
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    return positions[:N]

def construct():
    global _DEADLINE
    
    if time.time() > _DEADLINE - 10.0:
        # Fallback
        positions = _generate_initial_positions()
        angles = [0.0] * N
        verts = []
        for i in range(N):
            verts.extend(_get_hex_vertices(positions[i][0], positions[i][1], angles[i]))
        side, center, angle = _find_best_outer_lp(verts, 1.0)
        inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
        return inner, center, side, angle

    positions = _generate_initial_positions()
    angles = [0.0] * N
    
    # Optimization
    def objective(params):
        if time.time() > _DEADLINE - 5.0:
            return 1e10
            
        pos = [(params[2*i], params[2*i+1]) for i in range(N)]
        ang = list(params[2*N:2*N+N])
        
        # Check overlaps
        all_verts = []
        for i in range(N):
            all_verts.append(_get_hex_vertices(pos[i][0], pos[i][1], ang[i]))
            
        for i in range(N):
            for j in range(i+1, N):
                if _hexagons_overlap(all_verts[i], all_verts[j]):
                    return 1000.0 + i + j # Simple penalty
            
        # Flatten vertices for LP
        flat_verts = []
        for v in all_verts:
            flat_verts.extend(v)
            
        # Find best outer
        side, center, angle = _find_best_outer_lp(flat_verts, 0.5)
        
        if side == float('inf'):
            return 1e10
            
        return side

    x0 = []
    for p in positions:
        x0.extend([p[0], p[1]])
    x0.extend(angles)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success:
            pos = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            ang = list(res.x[2*N:2*N+N])
            
            # Verify
            all_verts = []
            valid = True
            for i in range(N):
                all_verts.append(_get_hex_vertices(pos[i][0], pos[i][1], ang[i]))
            for i in range(N):
                for j in range(i+1, N):
                    if _hexagons_overlap(all_verts[i], all_verts[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                flat_verts = []
                for v in all_verts:
                    flat_verts.extend(v)
                side, center, angle = _find_best_outer_lp(flat_verts, 1.0)
                inner = [[p[0], p[1], a] for p, a in zip(pos, ang)]
                return inner, center, side, angle
    except Exception as e:
        pass

    # Fallback if optimization failed or invalid
    all_verts = []
    for i in range(N):
        all_verts.append(_get_hex_vertices(positions[i][0], positions[i][1], angles[i]))
    flat_verts = []
    for v in all_verts:
        flat_verts.extend(v)
    side, center, angle = _find_best_outer_lp(flat_verts, 1.0)
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, center, side, angle
