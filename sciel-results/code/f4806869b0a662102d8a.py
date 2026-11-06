"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build a map from line key to list of point indices on that line."""
    n = len(points)
    line_map = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].append(i)
            line_map[key].append(j)
    # Deduplicate point indices per line
    for key in line_map:
        line_map[key] = list(set(line_map[key]))
    return line_map

def _evaluate(points):
    """
    Return (num_4_lines, is_valid) for the given points.
    num_4_lines: number of lines containing exactly 4 points.
    is_valid: True if no line contains 5 or more points.
    """
    n = len(points)
    line_map = _build_line_map(points)
    count_4 = 0
    valid = True
    for key, pts in line_map.items():
        if len(pts) >= 5:
            valid = False
            break
        if len(pts) == 4:
            count_4 += 1
    return count_4, valid

def _is_valid(points):
    _, valid = _evaluate(points)
    return valid

def _count_4_lines(points):
    count, _ = _evaluate(points)
    return count

def _ag24_points():
    """
    Generate 16 points from AG(2,4) affine plane embedded in integer lattice.
    We use a 4x4 grid as a starting point.
    """
    points = [[x, y] for x in range(4) for y in range(4)]
    return points

def _local_search(points, max_time=100.0, step_range=3):
    """
    Local search to improve the configuration.
    Returns (best_points, best_score).
    """
    start_time = time.time()
    limit = max_time
    best_points = [p[:] for p in points]
    best_score, valid = _evaluate(best_points)
    if not valid:
        # If invalid, try to fix by random moves
        for _ in range(1000):
            idx = random.randint(0, 15)
            nx = random.randint(-20, 20)
            ny = random.randint(-20, 20)
            collision = False
            for i in range(16):
                if i != idx and points[i][0] == nx and points[i][1] == ny:
                    collision = True
                    break
            if not collision:
                new_points = [p[:] for p in best_points]
                new_points[idx] = [nx, ny]
                _, v = _evaluate(new_points)
                if v:
                    best_points = new_points
                    best_score, _ = _evaluate(best_points)
                    break
        if not valid:
            # Still invalid, return as is
            return best_points, best_score

    while time.time() - start_time < limit:
        improved = False
        # Shuffle indices for diversity
        indices = list(range(16))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > limit - 2.0:
                break
            x, y = best_points[idx]
            # Try nearby moves with increasing step sizes
            for step in range(1, step_range + 1):
                if time.time() - start_time > limit - 1.0:
                    break
                for dx in range(-step, step + 1):
                    for dy in range(-step, step + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) > step or abs(dy) > step:
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        
                        # Check collision
                        collision = False
                        for i in range(16):
                            if i != idx and best_points[i][0] == nx and best_points[i][1] == ny:
                                collision = True
                                break
                        if collision:
                            continue
                        
                        new_points = [p[:] for p in best_points]
                        new_points[idx] = [nx, ny]
                        
                        # Check validity and score
                        score, valid = _evaluate(new_points)
                        if valid and score > best_score:
                            best_score = score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a random jump
            if random.random() < 0.15:
                idx = random.randint(0, 15)
                nx = random.randint(-25, 25)
                ny = random.randint(-25, 25)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    collision = False
                    for i in range(16):
                        if i != idx and best_points[i][0] == nx and best_points[i][1] == ny:
                            collision = True
                            break
                    if not collision:
                        new_points = [p[:] for p in best_points]
                        new_points[idx] = [nx, ny]
                        score, valid = _evaluate(new_points)
                        if valid and score > best_score:
                            best_score = score
                            best_points = new_points
                            improved = True
            if not improved:
                # Try larger random jumps
                if random.random() < 0.05:
                    idx = random.randint(0, 15)
                    nx = random.randint(-50, 50)
                    ny = random.randint(-50, 50)
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        collision = False
                        for i in range(16):
                            if i != idx and best_points[i][0] == nx and best_points[i][1] == ny:
                                collision = True
                                break
                        if not collision:
                            new_points = [p[:] for p in best_points]
                            new_points[idx] = [nx, ny]
                            score, valid = _evaluate(new_points)
                            if valid and score > best_score:
                                best_score = score
                                best_points = new_points
                                improved = True
                if not improved:
                    break

    return best_points, best_score

def _multi_start_search(max_time=100.0):
    """
    Multi-start local search with different initial configurations.
    """
    start_time = time.time()
    limit = max_time
    best_points = None
    best_score = -1
    
    # Start with AG(2,4) embedding (4x4 grid)
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted
    initial_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    
    # 4x4 grid scaled
    initial_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    
    # Random configurations
    for _ in range(5):
        pts = []
        for _ in range(16):
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 16:
            initial_configs.append(pts)
    
    # Try each initial configuration
    for config in initial_configs:
        if time.time() - start_time > limit - 10.0:
            break
        
        # Check if valid
        _, valid = _evaluate(config)
        if not valid:
            continue
        
        score, _ = _evaluate(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run local search
        remaining = limit - (time.time() - start_time) - 5.0
        if remaining > 5.0:
            pts, sc = _local_search(config, max_time=min(remaining, 30.0))
            if sc > best_score:
                best_score = sc
                best_points = pts
    
    return best_points, best_score

def construct():
    start_time = time.time()
    limit = 110.0  # Safety margin before 120s limit
    
    # Run multi-start search
    best_points, best_score = _multi_start_search(max_time=limit - 5.0)
    
    # Final check and return
    if best_points is None:
        # Fallback: return 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Ensure validity
    _, valid = _evaluate(best_points)
    if not valid:
        # Try to fix
        for _ in range(100):
            idx = random.randint(0, 15)
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            collision = False
            for i in range(16):
                if i != idx and best_points[i][0] == nx and best_points[i][1] == ny:
                    collision = True
                    break
            if not collision:
                new_points = [p[:] for p in best_points]
                new_points[idx] = [nx, ny]
                _, v = _evaluate(new_points)
                if v:
                    best_points = new_points
                    break
    
    return best_points
# EVOLVE-BLOCK-END
