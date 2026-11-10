import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 550

    def get_vertices(x, y, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack((x + np.cos(angles), y + np.sin(angles)))

    def check_separation_fast(v1, v2, c1, c2):
        # c1, c2 are centers (x, y)
        # Quick check: distance between centers
        dist = np.linalg.norm(c1 - c2)
        # Min distance between centers of non-intersecting hexagons is sqrt(3)
        # If dist > sqrt(3) + 0.05, they are likely separated
        if dist > 1.732 + 0.05:
            return False

        # Detailed check
        # Point in polygon
        def point_in_polygon(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1)%n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        for v in v1:
            if point_in_polygon(v, v2):
                return True
        for v in v2:
            if point_in_polygon(v, v1):
                return True

        def edges_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            e1 = (v1[i], v1[(i+1)%6])
            for j in range(6):
                e2 = (v2[j], v2[(j+1)%6])
                if edges_intersect(e1[0], e1[1], e2[0], e2[1]):
                    return True
        return False

    def objective(params):
        hexes = params.reshape(11, 3)
        
        vertices = []
        centers = []
        for i in range(11):
            x, y, a = hexes[i]
            vertices.append(get_vertices(x, y, a))
            centers.append((x, y))
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if check_separation_fast(vertices[i], vertices[j], centers[i], centers[j]):
                    penalty += 1.0
                    
        if penalty > 0:
            return penalty * 100.0

        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            normals.append((np.cos(angle), np.sin(angle)))
        
        max_proj = 0.0
        for v in vertices:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_proj:
                    max_proj = proj
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S

    # Initial Guess
    p0 = [0, 0, 0]
    r1 = 1.75
    ring1 = []
    for k in range(6):
        angle_deg = k * 60
        rad = np.deg2rad(angle_deg)
        x = r1 * np.cos(rad)
        y = r1 * np.sin(rad)
        ring1.append([x, y, 0])
        
    r2 = 3.5
    ring2 = []
    for angle_deg in [30, 150, 210, 330]:
        rad = np.deg2rad(angle_deg)
        x = r2 * np.cos(rad)
        y = r2 * np.sin(rad)
        ring2.append([x, y, 0])
        
    init_params = np.array(p0 + ring1 + ring2).flatten()

    best_result = None
    best_score = np.inf
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score and res.fun < 10:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass

    if best_result is None or best_score > 10:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = best_result.reshape(11, 3)
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_score
    
    if outer_hex_side_length < np.sqrt(11):
        outer_hex_side_length = np.sqrt(11)
        
    return inner_hex_data, outer_hex_data, outer_hex_side_length
