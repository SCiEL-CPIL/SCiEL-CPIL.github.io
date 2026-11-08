import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_polygon(pt, poly):
    """
    Check if point pt is inside polygon poly.
    """
    x, y = pt
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def polygons_intersect(v1, v2, eps=1e-9):
    """
    Check if two convex polygons intersect.
    Separating Axis Theorem (SAT).
    """
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
                axes.append(normal)
                
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
            
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons are disjoint and inside the outer hexagon.
    """
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    inner_verts_list = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts_list.append(v)
        
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False
                
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if polygons_intersect(inner_verts_list[i], inner_verts_list[j]):
                return False
                
    return True

def get_min_side_length_full(inner, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Calculate the minimum side length of a hexagon at outer_center/angle 
    that contains all inner hexagons.
    """
    max_proj = 0
    for x, y, ang in inner:
        verts = hexagon_vertices(x, y, 1.0, ang)
        for vx, vy in verts:
            dx = vx - outer_center[0]
            dy = vy - outer_center[1]
            
            theta = -np.radians(outer_angle)
            rx = dx * np.cos(theta) - dy * np.sin(theta)
            ry = dx * np.sin(theta) + dy * np.cos(theta)
            
            p1 = abs(rx)
            p2 = abs(0.5 * rx + np.sqrt(3)/2 * ry)
            p3 = abs(-0.5 * rx + np.sqrt(3)/2 * ry)
            
            current_max = max(p1, p2, p3)
            if current_max > max_proj:
                max_proj = current_max
                
    return 2 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    timeout = 115.0  # 150s limit, keep margin
    
    # Initial guess: Hexagonal packing
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (-1, 2), (-2, 1)
    ]
    
    inner = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    cm = np.mean(inner, axis=0)[:2]
    inner = [[x - cm[0], y - cm[1], a] for x, y, a in inner]
    
    def objective(params):
        inner_pos = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_pos.append([x, y, a])
            
        side = get_min_side_length_full(inner_pos)
        
        for i in range(N):
            v1 = hexagon_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_pos[i][2])
            for j in range(i+1, N):
                v2 = hexagon_vertices(inner_pos[j][0], inner_pos[j][1], 1.0, inner_pos[j][2])
                if polygons_intersect(v1, v2):
                    return side + 1000.0
            
        return side
        
    x0 = []
    for x, y, a in inner:
        x0.extend([x, y, a])
        
    best_side = get_min_side_length_full(inner)
    best_inner = inner
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'fatol': 1e-4, 'xatol': 1e-4})
        
        if res.fun < best_side:
            best_params = res.x
            best_inner = []
            for i in range(N):
                best_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
            best_side = res.fun
            
    except Exception:
        pass
        
    if check_validity(best_inner, (0.0, 0.0), best_side + 1e-6, 0.0):
        return best_inner, (0.0, 0.0), best_side, 0.0
    else:
        return inner, (0.0, 0.0), 100.0, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To strictly follow "Keep the function names and signatures of the seed program", 
# and since the seed program had construct inside the evolve block, 
# I will move the definition here or ensure it's the one called.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program:
# def construct():
#     ...
#     return ...
# 
# I will define construct inside the block to be safe, or just define it globally.
# The evaluator likely calls construct().
# I will place the main logic in a helper and have construct() call it, 
# or just define construct() directly.
# Given the structure of the seed, I'll put the implementation in the block.

def construct():
    # Use the global functions defined above
    start_time = time.time()
    timeout = 115.0
    
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (-1, 2), (-2, 1)
    ]
    
    inner = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    cm = np.mean(inner, axis=0)[:2]
    inner = [[x - cm[0], y - cm[1], a] for x, y, a in inner]
    
    def objective(params):
        inner_pos = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_pos.append([x, y, a])
            
        side = get_min_side_length_full(inner_pos)
        
        for i in range(N):
            v1 = hexagon_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_pos[i][2])
            for j in range(i+1, N):
                v2 = hexagon_vertices(inner_pos[j][0], inner_pos[j][1], 1.0, inner_pos[j][2])
                if polygons_intersect(v1, v2):
                    return side + 1000.0
            
        return side
        
    x0 = []
    for x, y, a in inner:
        x0.extend([x, y, a])
        
    best_side = get_min_side_length_full(inner)
    best_inner = inner
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'fatol': 1e-4, 'xatol': 1e-4})
        
        if res.fun < best_side:
            best_params = res.x
            best_inner = []
            for i in range(N):
                best_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
            best_side = res.fun
            
    except Exception:
        pass
        
    if check_validity(best_inner, (0.0, 0.0), best_side + 1e-6, 0.0):
        return best_inner, (0.0, 0.0), best_side, 0.0
    else:
        return inner, (0.0, 0.0), 100.0, 0.0

# EVOLVE-BLOCK-END
