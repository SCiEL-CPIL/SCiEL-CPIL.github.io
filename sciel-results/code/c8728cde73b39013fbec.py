"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    count = 0
    # For each pair of points, find the line, then count how many points are on it
    # To avoid counting the same line multiple times, use a set of normalized line representations
    lines_seen = set()
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the direction vector
            if dx == 0 and dy == 0:
                continue
            g = abs(dx)
            if g == 0:
                g = abs(dy)
            else:
                g = min(abs(dx), abs(dy)) if dx != 0 and dy != 0 else abs(dx) if dy == 0 else abs(dy)
            # Use gcd for proper normalization
            import math
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            nx = dx // g
            ny = dy // g
            
            # Ensure consistent sign: make the first non-zero component positive
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            # Line is determined by direction and one point on it
            # Use the point with smaller x, then smaller y as reference
            ref_x, ref_y = x1, y1
            if (x2, y2) < (x1, y1):
                ref_x, ref_y = x2, y2
            
            line_key = (nx, ny, ref_x, ref_y)
            
            if line_key in lines_seen:
                continue
            lines_seen.add(line_key)
            
            # Count points on this line
            cnt = 0
            for k in range(n):
                px, py = points[k]
                # Check if (px, py) is on the line through (ref_x, ref_y) with direction (nx, ny)
                # Cross product should be zero
                cross = (px - ref_x) * ny - (py - ref_y) * nx
                if cross == 0:
                    cnt += 1
            
            if cnt == 4:
                count += 1
    
    return count

def _no_five_collinear(points):
    """Check that no 5 points are collinear."""
    n = len(points)
    lines_seen = set()
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            import math
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            nx = dx // g
            ny = dy // g
            
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            ref_x, ref_y = x1, y1
            if (x2, y2) < (x1, y1):
                ref_x, ref_y = x2, y2
            
            line_key = (nx, ny, ref_x, ref_y)
            
            if line_key in lines_seen:
                continue
            lines_seen.add(line_key)
            
            cnt = 0
            for k in range(n):
                px, py = points[k]
                cross = (px - ref_x) * ny - (py - ref_y) * nx
                if cross == 0:
                    cnt += 1
                    if cnt >= 5:
                        return False
    
    return True

def _valid(points):
    """Check if configuration is valid: 16 distinct points, |x|,|y| <= 1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if not _no_five_collinear(points):
        return False
    return True

def _score(points):
    """Score: number of lines with exactly 4 points. Return -1 if invalid."""
    if not _valid(points):
        return -1
    return _count_lines_exactly_4(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    random.seed(42)
    
    # Start with 4x4 grid as initial solution
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Local search: try moving individual points
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = best_points[idx]
        
        # Try a random new position near the current one or random
        if random.random() < 0.5:
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        else:
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check if this position is already taken
        if any(p[0] == new_x and p[1] == new_y for p in best_points if p != best_points[idx]):
            continue
        
        # Try the move
        candidate = [p[:] for p in best_points]
        candidate[idx] = [new_x, new_y]
        
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_points = candidate
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
