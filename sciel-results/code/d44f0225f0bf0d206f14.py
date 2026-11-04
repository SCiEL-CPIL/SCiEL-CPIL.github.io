"""Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np

N = 12
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, center, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    cx, cy = center
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    s = side
    ax = abs(lx)
    ay = abs(ly)
    if ax > s:
        return False
    if ax <= s / 2:
        return ay <= s * math.sqrt(3) / 2
    else:
        return ay <= math.sqrt(3) * (s - ax)

def _hex_hex_overlap(v1, v2):
    """SAT check for two convex hexagons. Returns True if they overlap."""
    polys = [v1, v2]
    for i, poly in enumerate(polys):
        for j in range(len(poly)):
            x1, y1 = poly[j]
            x2, y2 = poly[(j + 1) % len(poly)]
            ex = x2 - x1
            ey = y2 - y1
            nx = -ey
            ny = ex
            proj1 = [px * nx + py * ny for (px, py) in v1]
            proj2 = [px * nx + py * ny for (px, py) in v2]
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            if max1 < min2 or max2 < min1:
                return False
    return True

def _check_validity(inner, outer_center, outer_side, outer_angle_deg, eps=1e-9):
    """Check if configuration is valid."""
    ocx, ocy = outer_center
    for (ix, iy, iang) in inner:
        iv = _hex_vertices((ix, iy), 1.0, iang)
        for (vx, vy) in iv:
            if not _point_in_hex(vx, vy, outer_center, outer_side, outer_angle_deg):
                return False
    n = len(inner)
    verts_list = [_hex_vertices((ix, iy), 1.0, iang) for (ix, iy, iang) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_hex_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def _make_honeycomb():
    positions = []
    s = 1.0
    dx = math.sqrt(3) * s
    dy = 1.5 * s
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    return positions

def _make_ring():
    positions = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        positions.append([x, y, 0.0])
    for i in range(5):
        angle = i * 2 * math.pi / 5 + math.pi / 5
        r = 2 * math.sqrt(3)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        positions.append([x, y, 0.0])
    return positions

def _make_strip():
    positions = []
    dx = math.sqrt(3)
    for i in range(12):
        positions.append([i * dx, 0.0, 0.0])
    return positions

def _optimize(inner, outer_center, outer_side, outer_angle_deg, time_budget):
    start_time = time.time()
    best_side = outer_side
    best_inner = [list(p) for p in inner]
    
    cur_inner = [list(p) for p in inner]
    cur_side = outer_side
    cur_center = list(outer_center)
    cur_angle = outer_angle_deg
    
    if _check_validity(cur_inner, cur_center, cur_side, cur_angle):
        best_inner = [list(p) for p in cur_inner]
        best_side = cur_side
    
    temp = 1.0
    cooling = 0.995
    min_temp = 1e-6
    
    while time.time() - start_time < time_budget and temp > min_temp:
        new_inner = [list(p) for p in cur_inner]
        new_side = cur_side
        new_center = list(cur_center)
        new_angle = cur_angle
        
        for _ in range(2):
            idx = np.random.randint(0, 37)
            if idx < 36:
                i = idx // 3
                param = idx % 3
                if param == 0:
                    new_inner[i][0] += np.random.normal(0, 0.1)
                elif param == 1:
                    new_inner[i][1] += np.random.normal(0, 0.1)
                else:
                    new_inner[i][2] += np.random.normal(0, 5.0)
            else:
                new_side += np.random.normal(0, 0.05)
        
        if _check_validity(new_inner, new_center, new_side, new_angle):
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(p) for p in new_inner]
            cur_inner = new_inner
            cur_side = new_side
        
        temp *= cooling
    
    return best_inner, cur_center, best_side, cur_angle

def construct():
    start_time = time.time()
    
    configs = [_make_honeycomb(), _make_ring(), _make_strip()]
    
    best_result = None
    best_score = float('inf')
    
    for cfg in configs:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        xs = [p[0] for p in cfg]
        ys = [p[1] for p in cfg]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        max_x = max(abs(x - cx) for x in xs) + 1
        max_y = max(abs(y - cy) for y in ys) + 1
        initial_side = max(max_x, max_y) * 1.5 + 2
        
        remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - start_time)
        if remaining < 1:
            break
        
        opt_inner, opt_center, opt_side, opt_angle = _optimize(
            cfg, (cx, cy), initial_side, 0.0, remaining
        )
        
        if _check_validity(opt_inner, opt_center, opt_side, opt_angle):
            if opt_side < best_score:
                best_score = opt_side
                best_result = (opt_inner, opt_center, opt_side, opt_angle)
    
    if best_result is None:
        inner = _make_honeycomb()
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        max_x = max(abs(x - cx) for x in xs) + 2
        max_y = max(abs(y - cy) for y in ys) + 2
        side = max(max_x, max_y) + 2
        best_result = (inner, (cx, cy), side, 0.0)
    
    return best_result
