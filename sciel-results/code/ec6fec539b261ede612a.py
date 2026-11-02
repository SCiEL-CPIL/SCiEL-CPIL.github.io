import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_hex_vertices(cx, cy, theta):
        angles = theta + np.arange(6) * (math.pi / 3.0)
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def point_hex_violation(px, py, cx, cy, S, beta_out=0.0):
        A = S * APOTHEM
        max_viol = 0.0
        for k in range(6):
            ang = beta_out + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            proj = (px - cx) * nx + (py - cy) * ny
            if proj > A:
                viol = proj - A
                if viol > max_viol:
                    max_viol = viol
            elif -proj > A:
                viol = -proj - A
                if viol > max_viol:
                    max_viol = viol
        return max_viol

    def point_in_hex(px, py, hex_verts):
        sign = 0
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
            if cross != 0:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def min_distance_hex_hex(v1, v2):
        # 1. Vertex in polygon
        for p in v1:
            if point_in_hex(p[0], p[1], v2):
                return -1.0
        for p in v2:
            if point_in_hex(p[0], p[1], v1):
                return -1.0
        
        # 2. Edge intersection
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j + 1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return -1.0
        
        # 3. Min distance (vertex to edge)
        min_dist = float('inf')
        # Distance from point to segment
        def dist_point_seg(p, a, b):
            # Project p onto line ab
            ab = b - a
            ap = p - a
            if np.dot(ab, ab) == 0:
                return np.linalg.norm(ap)
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        for p in v1:
            for j in range(6):
                a = v2[j]
                b = v2[(j + 1) % 6]
                d = dist_point_seg(p, a, b)
                if d < min_dist:
                    min_dist = d
        for p in v2:
            for i in range(6):
                a = v1[i]
                b = v1[(i + 1) % 6]
                d = dist_point_seg(p, a, b)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        cx = params[0]
        cy = params[1]
        S = params[2]
        
        pen_contain = 0.0
        hex_data = []
        for i in range(N):
            x_i = params[3 + i * 3]
            y_i = params[4 + i * 3]
            theta_i = params[5 + i * 3]
            theta_i = theta_i % (math.pi / 3.0)
            
            verts = get_hex_vertices(x_i, y_i, theta_i)
            hex_data.append(verts)
            
            for vx, vy in verts:
                v = point_hex_violation(vx, vy, cx, cy, S, 0.0)
                pen_contain += v * v
        
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_hex_hex(hex_data[i], hex_data[j])
                if d < 0.0:
                    pen_overlap += 10.0 * (1.0 + abs(d))
        
        return S + 50.0 * pen_contain + 100.0 * pen_overlap

    bounds = [(-3.0, 3.0), (-3.0, 3.0), (2.0, 6.0)]
    for i in range(N):
        bounds.append((-3.0, 3.0))
        bounds.append((-3.0, 3.0))
        bounds.append((0.0, math.pi / 3.0))

    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            tol=1e-5, 
            seed=42,
            polish=True,
            init='sobol',
            popsize=10
        )
        best_params = res.x
    except Exception:
        best_params = None

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 10.0, 0.0

    cx = best_params[0]
    cy = best_params[1]
    S = best_params[2]
    
    inner = []
    for i in range(N):
        x_i = best_params[3 + i * 3]
        y_i = best_params[4 + i * 3]
        theta_i = best_params[5 + i * 3]
        theta_deg = math.degrees(theta_i)
        inner.append([float(x_i), float(y_i), float(theta_deg)])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = 0.0

    return inner, center, side, rotation
