import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + math.cos(a), cy + math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2):
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def point_polygon_distance(pt, poly):
        min_dist = float('inf')
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                d = math.hypot(pt[0]-p1[0], pt[1]-p1[1])
            else:
                t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                proj_x = p1[0] + t*dx
                proj_y = p1[1] + t*dy
                d = math.hypot(pt[0]-proj_x, pt[1]-proj_y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def get_min_L(vertices_list):
        max_proj = 0.0
        normals = [math.radians(30 + 60*k) for k in range(6)]
        for v in vertices_list:
            for ang in normals:
                proj = v[0]*math.cos(ang) + v[1]*math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)

    def compute_L_and_penalty(params):
        positions = []
        vertices_list = []
        all_vertices = []
        
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            positions.append((x, y, th))
            verts = get_hex_vertices(x, y, th)
            vertices_list.append(verts)
            all_vertices.extend(verts)
            
        L_req = get_min_L(all_vertices)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = vertices_list[i]
                v2 = vertices_list[j]
                if not polygons_disjoint(v1, v2):
                    pen = 0.0
                    for v in v1:
                        if point_in_polygon(v, v2):
                            d = point_polygon_distance(v, v2)
                            pen += d * d
                    for v in v2:
                        if point_in_polygon(v, v1):
                            d = point_polygon_distance(v, v1)
                            pen += d * d
                    penalty += 10.0 + pen
                    
        return L_req, penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        L_req, penalty = compute_L_and_penalty(params)
        return L_req + penalty

    # Initializations
    grid_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y))
                
    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]
    
    params_grid = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        params_grid.extend([x, y, 0.0])
        
    np.random.seed(42)
    params_random = []
    for _ in range(N):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        th = np.random.uniform(0, 360)
        params_random.extend([x, y, th])

    best_result = None
    best_score = float('inf')
    best_params = None

    # Optimization with multiple restarts
    inits = [params_grid, params_random]
    
    # Add some more random inits with different seeds
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        pr = []
        for _ in range(N):
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(-2.5, 2.5)
            th = np.random.uniform(0, 360)
            pr.extend([x, y, th])
        inits.append(pr)

    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
            if time.time() < deadline:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # If best_params still has penalty, try a final polish with tighter tolerance
    if best_params is not None:
        L_req, penalty = compute_L_and_penalty(best_params)
        if penalty > 1e-6 and time.time() < deadline - 10.0:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x
            except Exception:
                pass

    if best_params is None:
        best_params = params_grid
        
    positions = []
    all_vertices = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        all_vertices.extend(verts)
        
    L_final = get_min_L(all_vertices)
    
    is_valid = True
    verts_list = [get_hex_vertices(pos[0], pos[1], pos[2]) for pos in positions]
    for i in range(N):
        for j in range(i+1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid: break
        
    if not is_valid:
        inner = []
        SCALE = 1.000001
        for i in init_grid_idx:
            x, y = grid_centers[i]
            inner.append([x * SCALE, y * SCALE, 0.0])
            
        fb_vertices = []
        for p in inner:
            v = get_hex_vertices(p[0], p[1], p[2])
            fb_vertices.extend(v)
        L_final = get_min_L(fb_vertices)
        
        return inner, (0.0, 0.0), L_final, 0.0

    return positions, (0.0, 0.0), L_final, 0.0
