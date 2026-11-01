import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 11
DEADLINE = 140.0

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    angle = np.radians(angle_deg)
    angles = angle + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_polygon(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hex_inside_hex(inner_hex, outer_hex):
    for vx, vy in inner_hex:
        if not point_in_polygon(vx, vy, outer_hex):
            return False
    return True

def check_disjoint(inner_hexes):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            h1 = inner_hexes[i]
            h2 = inner_hexes[j]
            # SAT: check if any edge normal separates them
            for poly in [h1, h2]:
                for k in range(6):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k+1) % 6]
                    dx = x2 - x1
                    dy = y2 - y1
                    nx, ny = -dy, dx
                    proj1 = np.dot(h1, [nx, ny])
                    proj2 = np.dot(h2, [nx, ny])
                    if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                        break
                else:
                    continue
                break
            else:
                return False
    return True

def construct():
    start_time = time.time()
    
    def make_honeycomb(rows):
        positions = []
        y = 0.0
        for i, count in enumerate(rows):
            x_start = -(count - 1) * np.sqrt(3) / 2
            for j in range(count):
                x = x_start + j * np.sqrt(3)
                if i % 2 == 1:
                    x += np.sqrt(3) / 2
                positions.append([x, y, 0.0])
            y += 1.5
        if positions:
            y_center = np.mean([p[1] for p in positions])
            x_center = np.mean([p[0] for p in positions])
            for p in positions:
                p[0] -= x_center
                p[1] -= y_center
        return positions
    
    row_patterns = [
        [3, 5, 3],
        [4, 3, 4],
        [2, 3, 3, 3],
        [5, 3, 3],
        [3, 4, 3, 1],
        [1, 3, 4, 3],
        [2, 4, 3, 2],
        [4, 4, 3],
        [3, 3, 3, 2],
        [2, 3, 4, 2],
        [3, 3, 5],
        [5, 3, 3],
        [4, 4, 3],
        [2, 4, 4, 1],
        [1, 4, 4, 2],
    ]
    
    configs = []
    for rows in row_patterns:
        if sum(rows) == N:
            configs.append(make_honeycomb(rows))
    
    best_result = None
    best_side = float('inf')
    
    for cfg_idx, inner in enumerate(configs):
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if inner hexagons are disjoint
        inner_hexes = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in inner]
        if not check_disjoint(inner_hexes):
            continue
        
        all_verts = np.vstack(inner_hexes)
        
        for outer_angle in [0, 30, 15, 45, 60, 75]:
            if time.time() - start_time > DEADLINE:
                break
            
            theta = np.radians(outer_angle)
            directions = theta + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
            cos_d = np.cos(directions)
            sin_d = np.sin(directions)
            
            max_vals = np.zeros(6)
            for k in range(6):
                proj = all_verts[:, 0] * cos_d[k] + all_verts[:, 1] * sin_d[k]
                max_vals[k] = np.max(proj)
            
            c = np.array([0.0, 0.0, 1.0])
            A_ub = np.zeros((6, 3))
            b_ub = np.zeros(6)
            for k in range(6):
                A_ub[k, 0] = -cos_d[k]
                A_ub[k, 1] = -sin_d[k]
                A_ub[k, 2] = -1.0
                b_ub[k] = -max_vals[k]
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            
            if res.success:
                cx, cy, t = res.x
                S = t / (np.sqrt(3) / 2)
                outer_hex = hexagon_vertices(cx, cy, S, outer_angle)
                valid = True
                for ih in inner_hexes:
                    if not hex_inside_hex(ih, outer_hex):
                        valid = False
                        break
                if valid and S < best_side:
                    best_side = S
                    best_result = (inner, (float(cx), float(cy)), float(S), float(outer_angle))
    
    if best_result is not None:
        return best_result
    
    # Fallback
    inner = configs[0]
    all_verts = []
    for pos in inner:
        verts = hexagon_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    x_min, y_min = np.min(all_verts, axis=0)
    x_max, y_max = np.max(all_verts, axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    S = max(x_max - x_min, y_max - y_min) / np.sqrt(3) + 1.0
    return inner, (float(cx), float(cy)), float(S), 0.0
