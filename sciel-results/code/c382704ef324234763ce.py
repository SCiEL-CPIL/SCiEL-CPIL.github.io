import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hexagon_vertices(center, side, angle_deg):
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xint = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x != p2x:
                        if x <= xint:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def _construct_impl():
    start_time = time.time()
    
    # Initial configuration: 7 central, 4 outer
    inner = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * 60
        rad = np.deg2rad(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        inner.append([x, y, 0.0])
        
    # 4 outer hexagons
    # Place at corners of the 7-hex cluster
    # Angles: 0, 90, 180, 270
    for i in range(4):
        angle = i * 90
        rad = np.deg2rad(angle)
        x = 2 * np.sqrt(3) * np.cos(rad)
        y = 2 * np.sqrt(3) * np.sin(rad)
        inner.append([x, y, 0.0])

    x0 = []
    for i in range(4):
        x0.extend(inner[7+i])
        
    def objective(params):
        current_inner = inner[:7].copy()
        for i in range(4):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            current_inner.append([x, y, a])
            
        penalty = 0.0
        polygons = []
        for h in current_inner:
            verts = _hexagon_vertices((h[0], h[1]), 1.0, h[2])
            polygons.append(verts)
            
        # Overlap penalty
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                # Simple distance check
                c1 = np.array([current_inner[i][0], current_inner[i][1]])
                c2 = np.array([current_inner[j][0], current_inner[j][1]])
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2 * 100.0
                # Vertex check for accuracy
                elif dist < 2.1:
                    if _polygons_overlap(polygons[i], polygons[j]):
                         penalty += 10.0

        # Max distance
        max_dist = 0.0
        for verts in polygons:
            d = np.max(np.linalg.norm(verts, axis=1))
            if d > max_dist:
                max_dist = d
                
        # Check containment in hexagon of side max_dist
        L = max_dist
        outer_verts = _hexagon_vertices((0.0, 0.0), L, 0.0)
        for verts in polygons:
            for p in verts:
                if not _point_in_polygon(p, outer_verts):
                    penalty += 50.0
                    
        return L + penalty

    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3})
    
    best_params = result.x
    best_inner = inner[:7].copy()
    for i in range(4):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner.append([x, y, a])
        
    max_dist = 0.0
    for h in best_inner:
        verts = _hexagon_vertices((h[0], h[1]), 1.0, h[2])
        d = np.max(np.linalg.norm(verts, axis=1))
        if d > max_dist:
            max_dist = d
            
    final_side = max_dist + 0.01
    return best_inner, (0.0, 0.0), final_side, 0.0

def _polygons_overlap(poly1, poly2):
    for p in poly1:
        if _point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if _point_in_polygon(p, poly1):
            return True
    return False

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END
