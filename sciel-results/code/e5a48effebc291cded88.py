import time
import random
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(pts):
    """Exact count of lines containing exactly two of the given points."""
    n = len(pts)
    line_pts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            # Canonical direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line: dy*x - dx*y = c
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    ordinary = 0
    for key, s in line_pts.items():
        if len(s) == 2:
            ordinary += 1
    return ordinary

def _valid(pts):
    if len(pts) != 12:
        return False
    seen = set()
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # not all collinear
    p0 = pts[0]
    for i in range(1, 12):
        p1 = pts[1]
        p2 = pts[i]
        if (p1[0]-p0[0])*(p2[1]-p0[1]) != (p1[1]-p0[1])*(p2[0]-p0[0]):
            return True
    return False

def _generate_conic_init(scale=1, offset_x=0, offset_y=0):
    """
    Generate 11 points on a parabola y = x^2 and 1 point off it.
    We choose parameters t for the points on the parabola.
    Point i: (t_i * scale + offset_x, (t_i * scale)^2 + offset_y)
    The 12th point is placed to intersect many secants.
    """
    pts = []
    # Choose 11 distinct integer parameters
    # Small integers give small coordinates
    params = list(range(1, 12)) 
    # Shuffle to vary the structure
    random.shuffle(params)
    
    for t in params:
        x = t * scale + offset_x
        y = (t * scale) ** 2 + offset_y
        pts.append([x, y])
        
    # Place the 12th point
    # A good candidate for a point intersecting many secants of a parabola 
    # is the "point at infinity" which we can't use, or a point on the axis of symmetry?
    # The axis of symmetry is x=0 (if centered).
    # Let's try placing it at (0, k) or similar.
    # Or better, search for a point that lies on many secants.
    # For now, place it at a random location near the center, then optimize.
    # A specific known configuration for 6 ordinary lines involves a specific arrangement.
    # Let's start with (0, 0) if not used, or (0, 1).
    
    # Ensure distinctness
    used = set((p[0], p[1]) for p in pts)
    candidates_for_12th = [
        [0, 0], [0, 1], [1, 0], [-1, 0], [0, -1],
        [offset_x, offset_y + 1], [offset_x + 1, offset_y]
    ]
    
    placed = False
    for c in candidates_for_12th:
        if (c[0], c[1]) not in used:
            pts.append(c)
            placed = True
            break
            
    if not placed:
        pts.append([offset_x + 10, offset_y + 10])
        
    return pts

def _local_search(initial_pts, time_limit):
    start_time = time.time()
    best_pts = initial_pts[:]
    best_val = _count_ordinary(best_pts)
    
    cur_pts = initial_pts[:]
    cur_val = best_val
    
    stall = 0
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 11)
        x, y = cur_pts[idx]
        
        # Generate candidates
        # 1. Random small move
        # 2. Move to a point on a line defined by two other points
        cands = set()
        
        # Random moves
        for _ in range(5):
            nx = x + random.randint(-10, 10)
            ny = y + random.randint(-10, 10)
            cands.add((nx, ny))
            
        # Line-based moves:
        # For every pair (j, k) != idx, find integer points on the line through j and k
        # that are close to cur_pts[idx]
        for j in range(12):
            if j == idx: continue
            for k in range(j+1, 12):
                if k == idx: continue
                x1, y1 = cur_pts[j]
                x2, y2 = cur_pts[k]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0: continue
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                # The line is (x1 + t*dx, y1 + t*dy)
                # We want t such that the point is close to (x, y)
                # (x - x1)/dx ~ (y - y1)/dy
                # Estimate t
                if dx != 0:
                    t_est = (x - x1) // dx
                else:
                    t_est = (y - y1) // dy
                    
                for t in range(t_est - 3, t_est + 4):
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Avoid duplicates
                    pt = (nx, ny)
                    # Check if pt is already in cur_pts (excluding idx)
                    is_dup = False
                    for m in range(12):
                        if m != idx and cur_pts[m][0] == nx and cur_pts[m][1] == ny:
                            is_dup = True
                            break
                    if not is_dup:
                        cands.add(pt)
                        
        # Evaluate candidates
        best_cand = None
        best_cand_val = cur_val
        
        for c in cands:
            new_pts = cur_pts[:]
            new_pts[idx] = list(c)
            if not _valid(new_pts):
                continue
            val = _count_ordinary(new_pts)
            if val < best_cand_val:
                best_cand_val = val
                best_cand = new_pts
        
        if best_cand is not None and best_cand_val < cur_val:
            cur_pts = best_cand
            cur_val = best_cand_val
            if cur_val < best_val:
                best_pts = cur_pts[:]
                best_val = cur_val
            stall = 0
        else:
            stall += 1
            if stall > 20:
                # Random restart of current point
                idx = random.randint(0, 11)
                cur_pts[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
                if _valid(cur_pts):
                    cur_val = _count_ordinary(cur_pts)
                stall = 0
                
    return best_pts, best_val

def _search():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_overall = None
    best_val_overall = float('inf')
    
    # Try multiple conic initializations
    configs = []
    for scale in [1, 2, 3]:
        for ox in [0, -5, 5]:
            for oy in [0, -10, 10]:
                configs.append(_generate_conic_init(scale, ox, oy))
                
    # Also try grid based
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Filter valid
    valid_configs = []
    for c in configs:
        if _valid(c):
            valid_configs.append(c)
            
    if not valid_configs:
        return [[x, y] for x in range(4) for y in range(3)]
        
    # Sort by initial score
    valid_configs.sort(key=lambda c: _count_ordinary(c))
    
    # Run local search on top configs
    # Allocate time
    time_per_search = 30.0
    remaining = deadline - time.time()
    
    for c in valid_configs[:3]: # Top 3
        if time.time() + time_per_search > deadline:
            break
        pts, val = _local_search(c, time_per_search)
        if val < best_val_overall:
            best_val_overall = val
            best_overall = pts
            
    if best_overall is None:
        best_overall = valid_configs[0]
        
    return best_overall

def construct():
    try:
        res = _search()
        if not _valid(res):
            res = [[x, y] for x in range(4) for y in range(3)]
        return res
    except Exception:
        return [[x, y] for x in range(4) for y in range(3)]
