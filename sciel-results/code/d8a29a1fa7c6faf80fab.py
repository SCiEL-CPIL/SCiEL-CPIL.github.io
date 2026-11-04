import numpy as np
import math

N = 12

def min_enclosing_hexagon_side_centers(points, angles):
    """
    Finds min side length S_centers of a regular hexagon enclosing the points.
    points: list of (x, y)
    angles: list of rotation angles in degrees
    """
    pts = np.array(points)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    pts = pts - np.array([cx, cy])
    
    min_S = float('inf')
    for ang_deg in angles:
        ang_rad = math.radians(ang_deg)
        base_angles = [math.radians(30), math.radians(90), math.radians(150), math.radians(210), math.radians(270), math.radians(330)]
        normals = []
        for ba in base_angles:
            a = ba + ang_rad
            normals.append([math.cos(a), math.sin(a)])
        normals = np.array(normals)
        
        proj = pts @ normals.T
        max_proj = np.max(proj, axis=0)
        r = np.max(max_proj)
        if r < 0: r = 0
        S_centers = (2.0 / math.sqrt(3.0)) * r
        if S_centers < min_S:
            min_S = S_centers
            
    return min_S

def construct():
    sqrt3 = math.sqrt(3)
    
    # Pattern 1: 3x4 Honeycomb
    pts1 = []
    for j in range(3):
        y = j * 1.5
        if j % 2 == 0:
            for i in range(4):
                pts1.append((i * sqrt3, y))
        else:
            for i in range(4):
                pts1.append(((i + 0.5) * sqrt3, y))
                
    # Pattern 2: 4x3 Honeycomb
    pts2 = []
    for j in range(4):
        y = j * 1.5
        if j % 2 == 0:
            for i in range(3):
                pts2.append((i * sqrt3, y))
        else:
            for i in range(3):
                pts2.append(((i + 0.5) * sqrt3, y))
                
    # Pattern 3: 2x6 Honeycomb
    pts3 = []
    for j in range(2):
        y = j * 1.5
        if j % 2 == 0:
            for i in range(6):
                pts3.append((i * sqrt3, y))
        else:
            for i in range(6):
                pts3.append(((i + 0.5) * sqrt3, y))
                
    # Pattern 4: Compact Cluster (7 + 5)
    # Center + 6 ring
    pts4 = [(0.0, 0.0)]
    for i in range(6):
        ang = math.radians(60 * i)
        pts4.append((sqrt3 * math.cos(ang), sqrt3 * math.sin(ang)))
    # Add 5 more. Let's try adding them to the right.
    # Rightmost is (sqrt3, 0).
    # Add (2*sqrt3, 0), (2.5*sqrt3, 1.5), (2.5*sqrt3, -1.5), (3*sqrt3, 0), (3.5*sqrt3, 1.5)?
    # This is getting complex. Let's stick to grids.
    
    angles = [0, 15, 30, 45, 60, 75, 90]
    
    best_S_total = float('inf')
    best_pts = None
    best_ang = 0
    
    for pts in [pts1, pts2, pts3]:
        S_centers = min_enclosing_hexagon_side_centers(pts, angles)
        S_total = S_centers + 1.0
        if S_total < best_S_total:
            best_S_total = S_total
            best_pts = pts
            # Find specific angle
            min_sc = float('inf')
            for a in angles:
                sc = min_enclosing_hexagon_side_centers(pts, [a])
                if sc < min_sc:
                    min_sc = sc
                    best_ang = a
                    
    if best_pts is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 40.0, 0.0
        
    cx = np.mean([p[0] for p in best_pts])
    cy = np.mean([p[1] for p in best_pts])
    inner = []
    for p in best_pts:
        inner.append([p[0] - cx, p[1] - cy, 0.0])
        
    return inner, (0.0, 0.0), best_S_total, best_ang
