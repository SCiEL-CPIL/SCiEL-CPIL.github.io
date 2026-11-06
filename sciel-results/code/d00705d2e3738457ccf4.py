import time
import random
from math import gcd

DEADLINE = 110

def _line_key(p1, p2):
    """Get a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    elif dy == 0:
        return ('h', y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _build_lines(points):
    """Build line -> set of point indices."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(lines):
    """Check no line has 4+ points."""
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _is_valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _construct_config():
    """
    Construct a configuration with 6 disjoint horizontal triples.
    This guarantees a matching number of at least 6, 
    implying the max no-three-collinear subset size is <= 14.
    """
    # Define 6 horizontal lines
    ys = [0, 10, 20, -10, -20, 30]
    
    # Define x-coordinates for each line to avoid vertical collisions
    # We need 3 distinct x's per line.
    # Total 18 points.
    # We want to avoid any x having 4 points.
    
    # Base x sets
    x_sets = [
        [-10, 0, 10],
        [-12, 2, 14],
        [-14, 4, 16],
        [-8, 1, 12],
        [-6, 3, 14],
        [-4, 5, 16]
    ]
    
    points = []
    for i, y in enumerate(ys):
        for x in x_sets[i]:
            points.append([x, y])
            
    # Add 2 more points to reach 20
    # Choose points that are unlikely to form 4-collinear
    # e.g. far away
    points.append([50, 50])
    points.append([-50, -50])
    
    return points

def _try_fix(points, rng):
    """
    Try to fix any 4-collinear violations by perturbing points.
    Returns a valid configuration or None.
    """
    points = [list(p) for p in points]
    for _ in range(100):
        lines = _build_lines(points)
        if _check_no_four_collinear(lines):
            return points
        
        # Find a line with 4+ points
        bad_line = None
        bad_pts = None
        for key, pts in lines.items():
            if len(pts) >= 4:
                bad_line = key
                bad_pts = list(pts)
                break
        
        if not bad_pts:
            return points
            
        # Pick one point from the bad line and move it
        idx_to_move = rng.choice(bad_pts)
        p = points[idx_to_move]
        
        # Try random new positions
        for _ in range(20):
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            
            # Ensure distinct
            if any(new_x == q[0] and new_y == q[1] for q in points):
                continue
                
            old_p = points[idx_to_move]
            points[idx_to_move] = [new_x, new_y]
            
            # Check validity
            if _is_valid(points):
                return points
            
            points[idx_to_move] = old_p
            
    return None

def construct():
    start_time = time.time()
    
    # 1. Try the constructive approach
    base_points = _construct_config()
    
    rng = random.Random(42)
    
    # Try to fix the base configuration
    result = _try_fix(base_points, rng)
    if result:
        return result
        
    # 2. If that fails, try random restarts with perturbed constructive base
    for i in range(1000):
        if time.time() - start_time > DEADLINE:
            break
            
        # Perturb the base points slightly
        perturbed = []
        for p in base_points:
            nx = p[0] + rng.randint(-2, 2)
            ny = p[1] + rng.randint(-2, 2)
            nx = max(-100, min(100, nx))
            ny = max(-100, min(100, ny))
            perturbed.append([nx, ny])
            
        # Ensure distinct
        seen = set()
        unique = []
        for p in perturbed:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique.append(p)
        
        while len(unique) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique.append([x, y])
                
        unique = unique[:20]
        
        if _is_valid(unique):
            return unique

    # Fallback: Random search
    while True:
        if time.time() - start_time > DEADLINE:
             # Return a simple valid set (parabola) - score will be high (20)
             return [[x, x*x] for x in range(-10, 10)]
        
        points = []
        while len(points) < 20:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in [tuple(p) for p in points]:
                points.append([x, y])
        if _is_valid(points):
            return points
