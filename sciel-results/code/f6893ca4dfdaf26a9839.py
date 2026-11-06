import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Count the number of ordinary lines (lines containing exactly 2 points).
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line key to set of point indices
    lines = defaultdict(set)
    
    # Generate all pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                # Should not happen for distinct points
                continue
            dx //= g
            dy //= g
            
            # Canonical form for direction: dx > 0, or dx == 0 and dy > 0
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            # Line equation: a*x + b*y + c = 0
            # Normal vector (a, b) is perpendicular to (dx, dy)
            # So a = dy, b = -dx
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _find_collinear_targets(points, max_k=5):
    """
    Find integer points that would be collinear with at least 2 existing points.
    These are good candidates for moving a point to, as they create 3-point lines.
    """
    n = len(points)
    targets = set()
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Check points p1 + k*(dx, dy) and p2 + k*(dx, dy)
            # We want points NOT already in the set
            # k should be small to keep coordinates reasonable and within bounds
            
            for k in range(-max_k, max_k + 1):
                if k == 0:
                    continue
                nx = x1 + k * dx
                ny = y1 + k * dy
                
                # Basic bounds check
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                # Check if this point is already in the configuration
                # To be fast, we don't check against all points, 
                # but the caller will handle duplicates.
                # However, we can check against the line's existing points if we tracked them.
                # For simplicity, just add to set.
                targets.add((nx, ny))
                
    return list(targets)

def _make_initial_configs():
    configs = []
    
    # 1. 3x4 Grid (Baseline)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 Grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 Grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. Diagonal Shift Grid (Sheared)
    # (x, y) -> (x+y, y)
    pts = []
    for y in range(4):
        for x in range(3):
            pts.append([x + y, y])
    configs.append(pts)
    
    # 6. 4x4 minus 4 corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    corners = [(0,0), (3,0), (0,3), (3,3)]
    pts = [p for p in pts if tuple(p) not in corners]
    configs.append(pts)
    
    # 7. Hexagonal-ish
    # Row 0: 0, 2, 4, 6
    # Row 1: 1, 3, 5, 7
    # Row 2: 0, 2, 4, 6
    pts = []
    for x in range(0, 8, 2):
        pts.append([x, 0])
    for x in range(1, 8, 2):
        pts.append([x, 1])
    for x in range(0, 8, 2):
        pts.append([x, 2])
    configs.append(pts)
    
    # 8. Random configs
    rng = random.Random(12345)
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            p = [rng.randint(-10, 10), rng.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        configs.append(pts)
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate initial configurations
    initial_configs = _make_initial_configs()
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if len(set(map(tuple, cfg))) < 12:
            continue
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
            
    if best_points is None:
        # Fallback to 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
        
    # Local Search
    rng = random.Random(42)
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_steps > max_no_improve:
            # Restart from a random configuration or a perturbed best
            # To avoid getting stuck, we can perturb the best solution significantly
            # or just accept that we might be at a local minimum.
            # Given the short time, we just continue with smaller steps or random restarts.
            # Let's do a random restart from a random point in the space
            # Actually, let's just reset no_improve and continue, maybe with different strategy weights
            no_improve_steps = 0

        strategy = rng.random()
        
        new_points = [p[:] for p in best_points]
        
        if strategy < 0.5:
            # Move one point randomly
            idx = rng.randint(0, 11)
            # Step size: small for fine tuning, large for exploration
            step = rng.choice([1, 2, 3, 5, 10, 20])
            dx = rng.randint(-step, step)
            dy = rng.randint(-step, step)
            if dx == 0 and dy == 0:
                continue
            new_points[idx] = [new_points[idx][0] + dx, new_points[idx][1] + dy]
            
        elif strategy < 0.8:
            # Move one point to a collinear target
            targets = _find_collinear_targets(best_points)
            if targets:
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                # Ensure target is not already the point itself
                if target[0] != new_points[idx][0] or target[1] != new_points[idx][1]:
                    new_points[idx] = list(target)
                else:
                    continue
            else:
                continue
                
        else:
            # Swap two points
            i, j = rng.sample(range(12), 2)
            new_points[i], new_points[j] = new_points[j], new_points[i]
            
        # Validate
        if len(set(map(tuple, new_points))) < 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in new_points):
            continue
            
        score = count_ordinary_lines(new_points)
        
        if score < best_score:
            best_score = score
            best_points = new_points
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            
        if best_score == 6:
            break
            
    return best_points
