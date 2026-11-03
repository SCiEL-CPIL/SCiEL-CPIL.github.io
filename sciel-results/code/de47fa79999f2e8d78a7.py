import numpy as np
from scipy.optimize import linprog
import time

DEADLINE = 140.0
START_TIME = time.time()

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def construct():
    global START_TIME, DEADLINE
    if time.time() - START_TIME > DEADLINE:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(11 // 2), 11 - 11 // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    s3 = np.sqrt(3)
    # Compact hexagonal arrangement of 11 unit hexagons
    positions = [
        (0.0, 0.0),
        (s3, 0.0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0.0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (s3, 1.5),
        (-s3, 1.5),
        (s3, -1.5),
        (-s3, -1.5),
    ]
    
    # Add small gap by scaling positions slightly
    gap_factor = 1.001
    positions = [(x * gap_factor, y * gap_factor) for x, y in positions]
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Get all vertices
    all_verts = []
    for x, y, a in inner:
        verts = hex_vertices(x, y, 1.0, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find minimum enclosing regular hexagon
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        angle_rad = np.radians(angle_deg)
        n_points = len(all_verts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A = []
        b = []
        for i in range(6):
            edge_angle = angle_rad + np.pi / 6 + i * np.pi / 3
            nx, ny = np.cos(edge_angle), np.sin(edge_angle)
            for px, py in all_verts:
                A.append([-nx, -ny, -np.sqrt(3)/2])
                b.append(-px*nx - py*ny)
        A.append([0.0, 0.0, -1.0])
        b.append(0.0)
        A = np.array(A)
        b = np.array(b)
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_center = (res.x[0], res.x[1])
                    best_angle = angle_deg
        except Exception:
            pass
    
    if best_side == float('inf'):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(11 // 2), 11 - 11 // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, best_center, best_side, best_angle
