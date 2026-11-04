import numpy as np
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    angles = angle + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_convex_poly(px, py, poly):
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < 0:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexagons_overlap(v1, v2):
    for i in range(6):
        if point_in_convex_poly(v1[i][0], v1[i][1], v2):
            return True
    for i in range(6):
        if point_in_convex_poly(v2[i][0], v2[i][1], v1):
            return True
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def min_hex_side_at_rotation(verts_all, cx, cy, theta_deg):
    theta = np.radians(theta_deg)
    R_needed = 0.0
    for k in range(6):
        angle = theta + np.radians(30 + 60 * k)
        nx, ny = np.cos(angle), np.sin(angle)
        for px, py in verts_all:
            proj = (px - cx) * nx + (py - cy) * ny
            if proj > 0:
                R_candidate = proj / (np.sqrt(3) / 2)
                if R_candidate > R_needed:
                    R_needed = R_candidate
    return R_needed

def min_hex_side(verts_all, cx, cy):
    best = np.inf
    for theta_deg in np.linspace(0, 60, 61):
        s = min_hex_side_at_rotation(verts_all, cx, cy, theta_deg)
        if s < best:
            best = s
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Initial: hexagonal lattice with spacing 2.1 (slightly more than 2 to avoid touching)
    positions = [[0.0, 0.0]]
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([2.1 * np.cos(a), 2.1 * np.sin(a)])
    for k in range(4):
        a = np.radians(60 * k + 30)
        positions.append([3.6 * np.cos(a), 3.6 * np.sin(a)])
    
    state = [[x, y, 0.0] for x, y in positions]
    
    def get_verts(st):
        return [hex_vertices(x, y, 1.0, a) for x, y, a in st]
    
    def all_verts(st):
        return np.vstack(get_verts(st))
    
    def valid(st):
        verts = get_verts(st)
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return False
        return True
    
    def cost(st):
        v = all_verts(st)
        cx, cy = np.mean(v[:,0]), np.mean(v[:,1])
        return min_hex_side(v, cx, cy)
    
    if not valid(state):
        for scale in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]:
            test = [[x*scale, y*scale, a] for x,y,a in state]
            if valid(test):
                state = test
                break
    
    best_state = [list(s) for s in state]
    best_cost = cost(state)
    
    T = 0.3
    cooling = 0.997
    
    while time.time() < deadline:
        new_state = [list(s) for s in state]
        idx = rng.integers(0, N)
        new_state[idx][0] += rng.normal(0, 0.25)
        new_state[idx][1] += rng.normal(0, 0.25)
        new_state[idx][2] += rng.normal(0, 8.0)
        
        if not valid(new_state):
            continue
        
        c = cost(new_state)
        delta = c - best_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 0.001)):
            state = new_state
            if c < best_cost:
                best_state = new_state[:]
                best_cost = c
        T *= cooling
    
    inner = [[float(s[0]), float(s[1]), float(s[2])] for s in best_state]
    v = all_verts(best_state)
    cx, cy = float(np.mean(v[:,0])), float(np.mean(v[:,1]))
    side = float(min_hex_side(v, cx, cy))
    
    # Find best rotation
    best_s = side
    best_th = 0.0
    for th in np.linspace(0, 60, 61):
        s = min_hex_side_at_rotation(v, cx, cy, th)
        if s < best_s:
            best_s = s
            best_th = th
    
    return inner, (cx, cy), float(best_s), float(best_th)
