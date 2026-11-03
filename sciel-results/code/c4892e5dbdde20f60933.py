import numpy as np
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0

def hex_vertices(cx, cy, angle_deg, size=1.0):
    rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + rad
    return np.column_stack((cx + size * np.cos(angles), cy + size * np.sin(angles)))

def point_in_convex_poly(pt, poly):
    for i in range(6):
        p1 = poly[i]
        p2 = poly[(i+1)%6]
        edge = p2 - p1
        vec = pt - p1
        cross = edge[0]*vec[1] - edge[1]*vec[0]
        if cross < -1e-9:
            return False
    return True

def is_valid_packing(inner, cx, cy, S, angle_outer):
    outer_verts = hex_vertices(cx, cy, angle_outer, S)
    for i in range(N):
        x, y, a = inner[i]
        iv = hex_vertices(x, y, a, 1.0)
        for pt in iv:
            if not point_in_convex_poly(pt, outer_verts):
                return False
    return True

def construct():
    sqrt3 = np.sqrt(3)
    
    # Candidate 1: 4x3
    inner1 = []
    for j in range(3):
        for i in range(4):
            x = i * sqrt3 + j * sqrt3/2
            y = j * 1.5
            inner1.append([x, y, 0.0])
    inner1 = np.array(inner1)
    
    # Candidate 2: 3x4
    inner2 = []
    for j in range(4):
        for i in range(3):
            x = i * sqrt3 + j * sqrt3/2
            y = j * 1.5
            inner2.append([x, y, 0.0])
    inner2 = np.array(inner2)
    
    # Candidate 3: 2x6
    inner3 = []
    for j in range(6):
        for i in range(2):
            x = i * sqrt3 + j * sqrt3/2
            y = j * 1.5
            inner3.append([x, y, 0.0])
    inner3 = np.array(inner3)
            
    candidates = [inner1, inner2, inner3]
    
    best_S = 100.0
    best_inner = None
    best_cx = 0
    best_cy = 0
    best_angle = 0
    
    for inner in candidates:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        cx = np.mean(inner[:, 0])
        cy = np.mean(inner[:, 1])
        
        for angle in np.linspace(0, 60, 13):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            lo = 3.0
            hi = 10.0
            for _ in range(25):
                mid = (lo + hi) / 2
                if is_valid_packing(inner, cx, cy, mid, angle):
                    hi = mid
                else:
                    lo = mid
            if hi < best_S:
                best_S = hi
                best_inner = inner
                best_cx = cx
                best_cy = cy
                best_angle = angle
                
    if best_inner is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner.tolist(), (best_cx, best_cy), best_S, best_angle
