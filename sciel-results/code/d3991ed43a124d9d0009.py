"""Structural change: GF(16)-inspired construction with structured local search."""
import time
import random
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_TIME_LIMIT = 110.0  # safety margin

def _time_left():
    if _START_TIME is None:
        return _TIME_LIMIT
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _score(points):
    """
    Returns (num_exact4_lines, valid).
    Uses bitmask to efficiently count points per line.
    """
    n = 16
    line_masks = {}
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            xj = xs[j]
            yj = ys[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            
            g = math.gcd(dx, dy)
            if g < 0:
                g = -g
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            a = dy_n
            b = -dx_n
            c = a * xi + b * yi
            
            g2 = math.gcd(abs(a), abs(b))
            if g2:
                a //= g2
                b //= g2
                c //= g2
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            
            mask = (1 << i) | (1 << j)
            if key in line_masks:
                line_masks[key] |= mask
            else:
                line_masks[key] = mask
                
    exact4 = 0
    for mask in line_masks.values():
        cnt = mask.bit_count()
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            return 0, False
            
    return exact4, True

def _inbounds(p):
    return -1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000

def _valid_config(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if not _inbounds(p):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    _, ok = _score(points)
    return ok

def _gf16_construction(scale=50, offset_x=0, offset_y=0):
    """
    Map 4x4 grid points to integers using a structured transformation
    inspired by GF(16) geometry. In GF(16), each line has exactly 4 points
    and no 5 are collinear. We approximate this structure over integers.
    
    Uses a nonlinear mapping to break grid symmetries while preserving
    the combinatorial structure of 4-point lines.
    """
    # Base 4x4 grid
    base = [(x, y) for x in range(4) for y in range(4)]
    
    # Apply a nonlinear warp to avoid obvious grid lines
    # Use quadratic terms to create "curved" lines that still have 4 points
    points = []
    for (x, y) in base:
        # Nonlinear transformation
        nx = scale * (x + x * x // 8) + offset_x
        ny = scale * (y + y * y // 8 + x // 2) + offset_y
        points.append([nx, ny])
    
    return points

def _structured_local_search(init_pts, rng, max_time):
    """
    Local search with structured moves:
    1. Single point moves (small perturbations)
    2. Pair swaps (exchange two points)
    3. Line-preserving moves (move a point along a direction that
       maintains certain collinearities)
    """
    cur = [list(p) for p in init_pts]
    cur_score, valid = _score(cur)
    if not valid:
        return None, -1
    
    best = [list(p) for p in cur]
    best_score = cur_score
    
    start = time.time()
    
    while time.time() - start < max_time:
        if _time_left() < 2.0:
            break
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single point move
            cand = [list(p) for p in cur]
            i = rng.randint(0, 15)
            dx = rng.randint(-10, 10)
            dy = rng.randint(-10, 10)
            new_p = [cand[i][0] + dx, cand[i][1] + dy]
            
            if not _inbounds(new_p):
                continue
            if any((new_p[0] == p[0] and new_p[1] == p[1]) and idx != i 
                   for idx, p in enumerate(cand)):
                continue
            cand[i] = new_p
            
        elif move_type < 0.8:
            # Pair swap
            cand = [list(p) for p in cur]
            i, j = rng.sample(range(16), 2)
            cand[i], cand[j] = cand[j], cand[i]
            
        else:
            # Larger random restart for a few points
            cand = [list(p) for p in cur]
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, 15)
                cand[i] = [rng.randint(-500, 500), rng.randint(-500, 500)]
        
        cs, valid = _score(cand)
        if not valid:
            continue
            
        # Accept if better, or with small probability if equal/worse
        if cs > cur_score:
            cur = cand
            cur_score = cs
            if cur_score > best_score:
                best_score = cur_score
                best = [list(p) for p in cur]
        elif cs == cur_score and rng.random() < 0.15:
            cur = cand
        elif cs < cur_score and rng.random() < 0.02:
            cur = cand
            cur_score = cs
    
    return best, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)
    
    best_pts = None
    best_score = -1
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. Simple 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    init_configs.append(grid)
    
    # 2. GF(16)-inspired constructions with different parameters
    for scale in [30, 50, 80, 120]:
        for ox in [-100, 0, 100]:
            for oy in [-100, 0, 100]:
                pts = _gf16_construction(scale, ox, oy)
                if _valid_config(pts):
                    init_configs.append(pts)
    
    # 3. Random configurations
    for _ in range(10):
        pts = []
        seen = set()
        while len(pts) < 16:
            x = rng.randint(-500, 500)
            y = rng.randint(-500, 500)
            if (x, y) not in seen:
                pts.append([x, y])
                seen.add((x, y))
        if _valid_config(pts):
            init_configs.append(pts)
    
    # Evaluate all initial configs and pick the best few
    scored_configs = []
    for cfg in init_configs:
        s, valid = _score(cfg)
        if valid:
            scored_configs.append((s, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Run local search on top configurations
    top_k = min(5, len(scored_configs))
    for i in range(top_k):
        if _time_left() < 5.0:
            break
        s, cfg = scored_configs[i]
        if s > best_score:
            best_score = s
            best_pts = [list(p) for p in cfg]
        
        # Allocate remaining time among searches
        remaining = _time_left()
        alloc = remaining / (top_k - i + 1)
        alloc = max(1.0, min(alloc, 30.0))
        
        result, rs = _structured_local_search(cfg, rng, alloc)
        if result is not None and rs > best_score:
            best_score = rs
            best_pts = result
    
    # If nothing valid, return a known valid config
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts
# EVOLVE-BLOCK-END
