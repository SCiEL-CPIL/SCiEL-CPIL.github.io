"""Packing 12 unit hexagons in a regular hexagon - minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
APOTHEM = np.sqrt(3) / 2  # apothem of unit hexagon

def hex_face_normals(theta):
    """6 outward face normals of a regular hexagon rotated by theta."""
    angles = theta + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, theta):
    """Return margin: positive if outside, negative if inside."""
    normals = hex_face_normals(theta)
    d = np.array([px - cx, py - cy])
    margins = normals @ d
    return np.max(margins) - side * APOTHEM

def hex_vertices_unit(theta=0.0):
    """Vertices of unit hexagon centered at origin."""
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def min_enclosing_hexagon(points):
    """
    Find minimum side length regular hexagon containing all points.
    Returns: (cx, cy, side, theta)
    """
    def side_for_theta(theta, points):
        normals = hex_face_normals(theta)
        projs = points @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        
        widths = np.zeros(3)
        for i in range(3):
            widths[i] = max_projs[i] + max_projs[i + 3]
        
        apothem_needed = np.max(widths) / 2
        side = apothem_needed / APOTHEM
        
        normals_3 = normals[:3]
        b = np.array([(max_projs[i] - max_projs[i + 3]) / 2 for i in range(3)])
        cx, _, _, _ = np.linalg.lstsq(normals_3, b, rcond=None)
        
        return side, cx
    
    best_side = np.inf
    best_params = None
    
    thetas = np.linspace(0, np.pi / 3, 90)
    for theta in thetas:
        side, cx = side_for_theta(theta, points)
        if side < best_side:
            best_side = side
            best_params = (cx[0], cx[1], side, theta)
    
    # Local refinement
    def obj(params):
        theta = params[0] % (np.pi / 3)
        cx, cy = params[1], params[2]
        normals = hex_face_normals(theta)
        d = points - np.array([cx, cy])
        margins = d @ normals.T
        max_projs = np.max(margins, axis=0)
        min_projs = np.min(margins, axis=0)
        widths = np.zeros(3)
        for i in range(3):
            widths[i] = max_projs[i] + max_projs[i + 3]
        apothem_needed = np.max(widths) / 2
        return apothem_needed / APOTHEM
    
    x0 = np.array([best_params[3], best_params[0], best_params[1]])
    try:
        result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-11})
        if result.fun < best_side:
            theta = result.x[0] % (np.pi / 3)
            side, cx = side_for_theta(theta, points)
            best_side = side
            best_params = (cx[0], cx[1], side, theta)
    except:
        pass
    
    return best_params

def check_validity(inner, center, side, angle):
    """Check containment and non-overlap."""
    cx, cy = center
    for (px, py, ptheta) in inner:
        if point_in_hexagon(px, py, cx, cy, side, angle) > 1e-9:
            return False
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            xi, yi, ti = inner[i]
            xj, yj, tj = inner[j]
            vi = hex_vertices_unit(ti)
            for v in vi:
                if point_in_hexagon(xi + v[0], yi + v[1], xj, yj, 1.0, tj) < 0:
                    return False
            vj = hex_vertices_unit(tj)
            for v in vj:
                if point_in_hexagon(xj + v[0], yj + v[1], xi, yi, 1.0, ti) < 0:
                    return False
    return True

def gen_two_rows():
    hexes = []
    for k in range(6):
        hexes.append([k * np.sqrt(3), 0.0, 0.0])
    for k in range(6):
        hexes.append([(k + 0.5) * np.sqrt(3), 1.5, 0.0])
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [[h[0] - cx, h[1] - cy, 0.0] for h in hexes]

def gen_compact():
    hexes = []
    pattern = [3, 4, 3, 2]
    for row, count in enumerate(pattern):
        for col in range(count):
            x = col * np.sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += np.sqrt(3) / 2
            hexes.append([x, y, 0.0])
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [[h[0] - cx, h[1] - cy, 0.0] for h in hexes]

def gen_grid_4x3():
    hexes = []
    for row in range(4):
        for col in range(3):
            x = col * np.sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += np.sqrt(3) / 2
            hexes.append([x, y, 0.0])
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [[h[0] - cx, h[1] - cy, 0.0] for h in hexes]

def gen_hex_cluster():
    """Center + 6 ring + 5 more."""
    hexes = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = k * np.pi / 3
        hexes.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle), 0.0])
    # Add 5 more from second ring
    for k in range(5):
        angle = k * 2 * np.pi / 5
        hexes.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle), 0.0])
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [[h[0] - cx, h[1] - cy, 0.0] for h in hexes]

def optimize_config(hexes, timeout=5.0):
    """Optimize hex positions to minimize enclosing hex side."""
    points = np.array([[h[0], h[1]] for h in hexes])
    cx, cy, side, theta = min_enclosing_hexagon(points)
    
    def objective(params):
        inner_xy = params[:2 * N].reshape(N, 2)
        cx_p, cy_p, side_p, theta_p = params[2 * N:2 * N + 4]
        
        # Containment penalty
        max_viol = 0
        for i in range(N):
            m = point_in_hexagon(inner_xy[i][0], inner_xy[i][1], cx_p, cy_p, side_p, theta_p)
            if m > 0:
                max_viol = max(max_viol, m)
        
        # Overlap penalty
        overlap = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(inner_xy[i] - inner_xy[j])
                if dist < np.sqrt(3) - 0.05:
                    overlap += (np.sqrt(3) - 0.05 - dist) ** 2
        
        return side_p + 50 * max_viol + 100 * overlap
    
    x0 = np.zeros(2 * N + 4)
    for i in range(N):
        x0[2 * i] = hexes[i][0]
        x0[2 * i + 1] = hexes[i][1]
    x0[2 * N:2 * N + 4] = [cx, cy, side, theta]
    
    bounds = [(-5, 5)] * 2 * N + [(0, 5), (0, 5), (1.5, 10), (0, np.pi / 3)]
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-12})
        if result.fun < side + 0.5:
            inner_xy = result.x[:2 * N].reshape(N, 2)
            new_hexes = [[inner_xy[i][0], inner_xy[i][1], 0.0] for i in range(N)]
            pts = np.array([[h[0], h[1]] for h in new_hexes])
            cx_f, cy_f, side_f, theta_f = min_enclosing_hexagon(pts)
            return new_hexes, (cx_f, cy_f), side_f, theta_f
    except:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_side = np.inf
    best_result = None
    
    generators = [gen_two_rows, gen_compact, gen_grid_4x3, gen_hex_cluster]
    
    for gen in generators:
        if time.time() > deadline - 15:
            break
        try:
            hexes = gen()
        except:
            continue
        if len(hexes) != N:
            continue
        
        points = np.array([[h[0], h[1]] for h in hexes])
        cx, cy, side, theta = min_enclosing_hexagon(points)
        
        if check_validity(hexes, (cx, cy), side, theta):
            if side < best_side:
                best_side = side
                best_result = ([list(h) for h in hexes], (cx, cy), side, theta)
        
        if time.time() < deadline - 20:
            opt_res = optimize_config(hexes)
            if opt_res:
                oh, oc, os, ot = opt_res
                if check_validity(oh, oc, os, ot):
                    if os < best_side:
                        best_side = os
                        best_result = (oh, oc, os, ot)
    
    if best_result is None:
        hexes = gen_two_rows()
        points = np.array([[h[0], h[1]] for h in hexes])
        cx, cy, side, theta = min_enclosing_hexagon(points)
        side *= 1.1
        best_result = ([list(h) for h in hexes], (cx, cy), side, theta)
    
    return best_result

# EVOLVE-BLOCK-START
# construct() is defined above
# EVOLVE-BLOCK-END
