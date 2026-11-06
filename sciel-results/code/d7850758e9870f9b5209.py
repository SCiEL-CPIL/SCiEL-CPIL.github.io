"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Normalize sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line: dx*x + dy*y + c = 0
    c = -(dx * y1 + dy * x1)
    # Normalize gcd(dx, dy, c)
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _eval_config(points):
    """Return (num_ordinary, valid) for a configuration of 12 points."""
    if len(points) != 12:
        return (999, False)
    if len(set(tuple(p) for p in points)) != 12:
        return (999, False)
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (999, False)
    if not _not_all_collinear(points):
        return (999, False)
    return (_count_ordinary(points), True)

def _local_search(points, deadline):
    """Local search: move individual points to reduce ordinary lines."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    # Precompute line counts for efficiency
    def _get_line_counts(pts):
        line_counts = {}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                key = _line_key(pts[i], pts[j])
                if key is not None:
                    line_counts[key] = line_counts.get(key, 0) + 1
        return line_counts
    
    current_line_counts = _get_line_counts(current)
    
    while time.time() < deadline:
        improved = False
        # Try moving each point
        for idx in range(12):
            old_x, old_y = current[idx]
            
            # Try nearby moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check duplicate
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    old_pt = current[idx]
                    current[idx] = [new_x, new_y]
                    
                    if not _not_all_collinear(current):
                        current[idx] = old_pt
                        continue
                    
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        current_line_counts = _get_line_counts(current)
                        break
                    else:
                        current[idx] = old_pt
                else:
                    continue
                break
            if improved:
                break
        
        if not improved:
            # Random restart within neighborhood
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            improved = False
            for _ in range(30):
                new_x = old_x + random.randint(-8, 8)
                new_y = old_y + random.randint(-8, 8)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                old_pt = current[idx]
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = old_pt
                    continue
                new_score = _count_ordinary(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    current_line_counts = _get_line_counts(current)
                    break
                else:
                    current[idx] = old_pt
            if not improved:
                break
    
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    random.seed(42)
    
    best_config = None
    best_score = float('inf')
    
    # Multiple initial configurations
    initial_configs = []
    
    # 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Two rows of 6
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Three rows of 4 (same as 3x4 but explicit)
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Structured: points on 3 lines with 4 points each, arranged to maximize intersections
    # Line y=0: (0,0),(2,0),(4,0),(6,0)
    # Line y=2: (0,2),(2,2),(4,2),(6,2)
    # Line y=4: (0,4),(2,4),(4,4),(6,4)
    # But this is just a 4x3 grid again.
    
    # Try a configuration with more collinearities
    # 4 points on x-axis: (0,0),(1,0),(2,0),(3,0)
    # 4 points on y-axis: (0,1),(0,2),(0,3),(0,4) -- but (0,0) is shared
    # So we have 7 distinct points so far. Add 5 more.
    # Let's try: (0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(0,4),(1,1),(2,2),(3,3),(1,2)
    # That's 12 points. Let's check.
    initial_configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[0,2],[0,3],[0,4],
        [1,1],[2,2],[3,3],[1,2]
    ])
    
    # Another structured config: 3 groups of 4 on 3 parallel lines, offset
    # y=0: (0,0),(1,0),(2,0),(3,0)
    # y=1: (1,1),(2,1),(3,1),(4,1)
    # y=2: (2,2),(3,2),(4,2),(5,2)
    initial_configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [1,1],[2,1],[3,1],[4,1],
        [2,2],[3,2],[4,2],[5,2]
    ])
    
    # Random configs
    for _ in range(30):
        pts = []
        while len(pts) < 12:
            x = random.randint(-12, 12)
            y = random.randint(-12, 12)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            initial_configs.append(pts)
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        score, valid = _eval_config(cfg)
        if valid and score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    # Local search from best initial
    if best_config:
        best_config, best_score = _local_search(best_config, deadline)
    
    # Try local search from other good configs if time permits
    for cfg in initial_configs:
        if time.time() > deadline - 5:
            break
        score = _count_ordinary(cfg)
        if score <= best_score + 3:  # Only try promising ones
            cfg_best, cfg_score = _local_search(cfg, deadline)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    return [[p[0], p[1]] for p in best_config]
# EVOLVE-BLOCK-END
