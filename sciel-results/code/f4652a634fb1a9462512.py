"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(points):
    """Count ordinary lines."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check collinearity
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _generate_initial_configs():
    """Generate a variety of initial configurations."""
    configs = []
    
    # 1. 3x4 Grid
    grid_3x4 = [(i, j) for i in range(3) for j in range(4)]
    configs.append(grid_3x4)
    
    # 2. 4x3 Grid
    grid_4x3 = [(i, j) for i in range(4) for j in range(3)]
    configs.append(grid_4x3)
    
    # 3. 6x2 Grid
    grid_6x2 = [(i, j) for i in range(6) for j in range(2)]
    configs.append(grid_6x2)
    
    # 4. 2x6 Grid
    grid_2x6 = [(i, j) for i in range(2) for j in range(6)]
    configs.append(grid_2x6)
    
    # 5. 4x4 Grid minus 4 corners
    grid_4x4 = [(i, j) for i in range(4) for j in range(4)]
    # Remove (0,0), (0,3), (3,0), (3,3)
    pts = [p for p in grid_4x4 if not (p[0] in [0,3] and p[1] in [0,3])]
    configs.append(pts)
    
    # 6. 4x4 Grid minus 4 specific points to form diagonals
    # Try removing (0,0), (1,1), (2,2), (3,3)
    pts = [p for p in grid_4x4 if p[0] != p[1]]
    configs.append(pts)
    
    # 7. Cyclic configuration on a circle (approximated with integers)
    # Use points on a parabola y=x^2
    pts = [(i, i*i) for i in range(12)]
    configs.append(pts)
    
    # 8. Random points in a box
    random.seed(42)
    for _ in range(10):
        pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        if _is_valid(pts):
            configs.append(pts)
            
    return configs

def _local_search(points, deadline):
    """
    Local search to minimize ordinary lines.
    Moves points to positions on existing lines or nearby integer coordinates.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Pre-calculate lines for initial config to build candidate pool
    def get_candidate_pool(pts):
        pool = set()
        # Add points on lines defined by pairs of points
        for i in range(12):
            for j in range(i+1, 12):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                # Extend line
                for t in range(-5, 6):
                    nx = x1 + t*dx
                    ny = y1 + t*dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        pool.add((nx, ny))
        # Add nearby integer points
        for p in pts:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    pool.add((p[0]+dx, p[1]+dy))
        return list(pool)
    
    candidate_pool = get_candidate_pool(current)
    random.shuffle(candidate_pool)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_pool)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_pool:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    # Update candidate pool occasionally
                    if no_improve == 0:
                        candidate_pool = get_candidate_pool(current)
                        random.shuffle(candidate_pool)
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140
    
    configs = _generate_initial_configs()
    
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if time.time() >= deadline - 5:
            break
        if not _is_valid(cfg):
            continue
        s = _count_ordinary(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
        
        # Local search from this config
        if time.time() < deadline - 10:
            ls_points, ls_score = _local_search(cfg, deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points
        
        if best_score <= 6:
            break
            
    if best_points is None:
        # Fallback
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
        
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
