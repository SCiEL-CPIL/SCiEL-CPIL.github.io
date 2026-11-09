import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(angle_rad + k * np.pi / 3),
                      cy + side * np.sin(angle_rad + k * np.pi / 3)] for k in range(6)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    apothem = side * np.sqrt(3) / 2
    angle_rad = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        if dx * np.cos(theta) + dy * np.sin(theta) > apothem + 1e-9:
            return False
    return True

def check_valid(inner, center, side, angle_deg):
    cx, cy = center
    for (ix, iy, ia) in inner:
        for vx, vy in hexagon_vertices(ix, iy, 1.0, ia):
            if not point_in_hexagon(vx, vy, cx, cy, side, angle_deg):
                return False
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            for vx, vy in hexagon_vertices(ix1, iy1, 1.0, ia1):
                if point_in_hexagon(vx, vy, ix2, iy2, 1.0, ia2):
                    return False
            for vx, vy in hexagon_vertices(ix2, iy2, 1.0, ia2):
                if point_in_hexagon(vx, vy, ix1, iy1, 1.0, ia1):
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0
    sqrt3 = np.sqrt(3)
    
    # Compact honeycomb cluster of 11 hexagons
    positions = []
    for j in [-1, 0, 1]:
        positions.append((j * sqrt3, 0.0, 0.0))
    for j in [-2, -1, 0, 1]:
        positions.append((j * sqrt3 + sqrt3/2, 1.5, 0.0))
    for j in [-1, 0, 1]:
        positions.append((j * sqrt3, 3.0, 0.0))
    positions.append((sqrt3/2, 4.5, 0.0))
    
    inner = [[x, y, a] for (x, y, a) in positions]
    cx = float(np.mean([p[0] for p in inner]))
    cy = float(np.mean([p[1] for p in inner]))
    
    max_dist = max(np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) + 1.0 for p in inner)
    init_side = max_dist * 2 / sqrt3 + 0.5
    best = (inner, (cx, cy), init_side, 0.0)
    
    def objective(params):
        c_x, c_y, s, a = params
        if s < 0.1:
            return 1e6
        apothem = s * np.sqrt(3) / 2
        a_rad = np.radians(a)
        viol = 0.0
        for (ix, iy, ia) in inner:
            for vx, vy in hexagon_vertices(ix, iy, 1.0, ia):
                dx, dy = vx - c_x, vy - c_y
                for k in range(6):
                    th = a_rad + k * np.pi / 3
                    v = dx * np.cos(th) + dy * np.sin(th)
                    if v > apothem:
                        viol = max(viol, v - apothem)
        return s + 100.0 * viol
    
    for ang0 in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        if time.time() > deadline - 15:
            break
        x0 = [cx, cy, init_side, ang0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-9})
            c_x, c_y, s, a = res.x
            if s > 0.1 and check_valid(inner, (c_x, c_y), s, a):
                if s < best[2]:
                    best = (inner, (c_x, c_y), s, a)
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
