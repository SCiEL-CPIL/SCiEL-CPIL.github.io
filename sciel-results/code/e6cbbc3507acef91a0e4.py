import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(center, angle_deg):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + theta
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def get_outer_hex_vertices(S, angle_deg=0.0):
    cx, cy = 0.0, 0.0
    theta = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + theta
    xs = cx + S * np.cos(angles)
    ys = cy + S * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_polygon(point, poly):
    x, y = point
    n = len(poly)
    sign = None
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        px = x - x1
        py = y - y1
        cross = dx * py - dy * px
        if sign is None:
            if cross == 0:
                continue
            sign = cross > 0
        else:
            if cross == 0:
                continue
            if (cross > 0) != sign:
                return False
    return True

def min_distance_polygon_polygons(p1, p2):
    min_dist = float('inf')
    for v in p1:
        if point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if point_in_polygon(v, p1):
            return 0.0
            
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        v1 = p1[i]
        v2 = p1[(i + 1) % n1]
        for j in range(n2):
            v3 = p2[j]
            v4 = p2[(j + 1) % n2]
            def dist_point_segment(p, a, b):
                ab = b - a
                ap = p - a
                ab_len_sq = np.dot(ab, ab)
                if ab_len_sq == 0:
                    return np.linalg.norm(ap)
                t = np.dot(ap, ab) / ab_len_sq
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            d1 = dist_point_segment(v1, v3, v4)
            d2 = dist_point_segment(v2, v3, v4)
            d3 = dist_point_segment(v3, v1, v2)
            d4 = dist_point_segment(v4, v1, v2)
            min_dist = min(min_dist, d1, d2, d3, d4)
            
    return min_dist

def check_overlap_penalty(hexes, outer_S):
    penalty = 0.0
    verts = [get_hex_vertices((h[0], h[1]), h[2]) for h in hexes]
    outer_verts = get_outer_hex_vertices(outer_S)
    
    for i in range(N):
        for v in verts[i]:
            if not point_in_polygon(v, outer_verts):
                min_d = float('inf')
                for j in range(6):
                    e1 = outer_verts[j]
                    e2 = outer_verts[(j+1)%6]
                    ab = e2 - e1
                    ap = v - e1
                    ab_len_sq = np.dot(ab, ab)
                    if ab_len_sq == 0: continue
                    t = np.dot(ap, ab) / ab_len_sq
                    t = max(0, min(1, t))
                    proj = e1 + t * ab
                    d = np.linalg.norm(v - proj)
                    min_d = min(min_d, d)
                penalty += min_d * 10.0
                
    for i in range(N):
        for j in range(i + 1, N):
            d = min_distance_polygon_polygons(verts[i], verts[j])
            if d < 0.05:
                penalty += max(0, 0.05 - d) * 100.0

    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    hexes = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = k * 60
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        hexes.append([x, y, 0.0])
    for k in range(4):
        angle = 30 + k * 90
        rad = np.deg2rad(angle)
        x = 2.5 * np.cos(rad)
        y = 2.5 * np.sin(rad)
        hexes.append([x, y, 0.0])
        
    lo = 3.0
    hi = 5.0
    best_S = hi
    best_hexes = hexes
    
    while time.time() < deadline:
        mid = (lo + hi) / 2.0
        x0 = np.array([val for h in hexes for val in h])
        
        def objective(params):
            h = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                a = params[3*i+2]
                h.append([x, y, a])
            return check_overlap_penalty(h, mid)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < 0.1:
                best_S = mid
                best_hexes = []
                for i in range(N):
                    best_hexes.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
                hi = mid
                hexes = best_hexes
            else:
                lo = mid
        except Exception:
            lo = mid

    return best_hexes, (0.0, 0.0), best_S, 0.0
