"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # safety margin before 150s kill


def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)


def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(seen) < 3:
        return False
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    # Cross product
    if (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) != 0:
        return True
    # Check all points collinear with p0, p1
    for i in range(3, 12):
        pi = points[i]
        if (p1[0] - p0[0]) * (pi[1] - p0[1]) - (p1[1] - p0[1]) * (pi[0] - p0[0]) != 0:
            return True
    return False


def _grid_config(rows, cols):
    """Generate a grid configuration."""
    pts = []
    for x in range(cols):
        for y in range(rows):
            pts.append([x, y])
    return pts[:12]


def _construct_hesse_based():
    """
    Strategy: Start with the 3x3 grid (9 points, 0 ordinary lines).
    Add 3 points to minimize new ordinary lines.
    The 3x3 grid points are (0,0)..(2,2).
    We search for 3 additional points P1, P2, P3.
    To reduce ordinary lines, P_i should be collinear with pairs of grid points.
    Grid lines have slopes 0, inf, 1, -1 (in 3x3, slopes are 0, inf, 1, -1? 
    (0,0)-(1,1) slope 1. (0,1)-(1,0) slope -1.
    (0,0)-(2,1) slope 1/2.
    There are many lines.
    We will use a local search to place the 3 extra points.
    """
    base_pts = [[0, 0], [1, 0], [2, 0],
                [0, 1], [1, 1], [2, 1],
                [0, 2], [1, 2], [2, 2]]
    
    # Initial guess for extra points: far away
    # But we want them to create collinearities.
    # Let's try specific structured positions first.
    # The Hesse configuration extra points are often related to the directions.
    # Try points at (10, 10), (10, -10), (-10, 10) etc.
    
    best_pts = base_pts + [[10, 10], [10, -10], [-10, 10]]
    best_count = count_ordinary_lines(best_pts)
    
    # Local search on the 3 extra points
    # We will perturb the 3 extra points
    current = [p[:] for p in best_pts]
    current_count = best_count
    
    # Define a set of candidate "anchor" positions or just random search
    # Random search with bias towards grid lines
    
    for _ in range(50000):
        if time.time() > _DEADLINE - 10:
            break
        
        # Pick one of the 3 extra points to move
        idx = random.randint(9, 11)
        
        # Strategy: Move to a point that lies on a line defined by 2 grid points
        # or 2 other points.
        # Generate a random line through 2 grid points, and pick a random integer point on it.
        if random.random() < 0.7:
            # Pick 2 grid points
            g1_idx = random.randint(0, 8)
            g2_idx = random.randint(0, 8)
            if g1_idx == g2_idx:
                continue
            p1 = base_pts[g1_idx]
            p2 = base_pts[g2_idx]
            
            # Line through p1, p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            
            # Parametric: (x1 + t*dx, y1 + t*dy)
            # Choose t such that point is not in base_pts and within bounds
            t = random.randint(-50, 50)
            if t == 0 or t == 1:
                continue
            new_x = p1[0] + t * dx
            new_y = p1[1] + t * dy
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            new_pt = [new_x, new_y]
            
            # Check if it collides with existing points
            if new_pt in [p[:] for p in current]:
                continue
            
            current[idx] = new_pt
        else:
            # Random perturbation
            current[idx][0] += random.randint(-20, 20)
            current[idx][1] += random.randint(-20, 20)
            
            if abs(current[idx][0]) > 10**9 or abs(current[idx][1]) > 10**9:
                current[idx][0] = 0
                current[idx][1] = 0
            
            if current[idx] in [p[:] for p in current[:idx]] + [p[:] for p in current[idx+1:]]:
                continue

        if not _is_valid(current):
            continue
            
        cnt = count_ordinary_lines(current)
        if cnt < current_count:
            current_count = cnt
            if cnt < best_count:
                best_count = cnt
                best_pts = [p[:] for p in current]
                
    return best_pts, best_count


def construct():
    """Find 12 integer points minimizing ordinary lines."""
    # Start with the Hesse-based construction
    best_pts, best_count = _construct_hesse_based()
    
    # Fallback to 3x4 grid if something goes wrong
    if not _is_valid(best_pts):
        best_pts = _grid_config(3, 4)
        best_count = count_ordinary_lines(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
