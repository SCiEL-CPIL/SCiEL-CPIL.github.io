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


def _regular_polygon(n, radius=1000):
    """Generate n points on a regular polygon (approximate integer)."""
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = int(round(radius * math.cos(angle)))
        y = int(round(radius * math.sin(angle)))
        pts.append([x, y])
    # Ensure distinctness
    seen = set()
    unique = []
    for p in pts:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _grid_config(rows, cols):
    """Generate a grid configuration."""
    pts = []
    for x in range(cols):
        for y in range(rows):
            pts.append([x, y])
    return pts[:12]


def _cubic_curve_points():
    """Points on a cubic curve (like y^2 = x^3 + x + 1) or similar."""
    # Try to use points from a known configuration
    # The 6-ordinary-line configuration for 12 points is related to
    # the Hesse configuration or similar
    # For now, try a few structured configurations
    return []


def _random_grid_perturb(base, max_perturb=100):
    """Perturb a grid configuration randomly."""
    pts = [p[:] for p in base]
    for i in range(len(pts)):
        if random.random() < 0.3:
            pts[i][0] += random.randint(-max_perturb, max_perturb)
            pts[i][1] += random.randint(-max_perturb, max_perturb)
    return pts


def _search_grid_configs(best_pts, best_count, time_limit):
    """Search over various grid and structured configurations."""
    configs = []
    # Standard grids
    for rows in range(1, 7):
        for cols in range(1, 7):
            if rows * cols >= 12:
                configs.append(_grid_config(rows, cols))
    # Regular polygons
    for radius in [100, 500, 1000, 5000, 10000]:
        poly = _regular_polygon(12, radius)
        if len(poly) == 12:
            configs.append(poly)
    # Some structured configurations
    # 3x4 grid with one point moved
    base = _grid_config(3, 4)
    for i in range(12):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                perturbed = [p[:] for p in base]
                perturbed[i][0] += dx
                perturbed[i][1] += dy
                configs.append(perturbed)
    
    for cfg in configs:
        if time.time() > time_limit:
            break
        if not _is_valid(cfg):
            continue
        cnt = count_ordinary_lines(cfg)
        if cnt < best_count:
            best_count = cnt
            best_pts = [p[:] for p in cfg]
    return best_pts, best_count


def _random_search(best_pts, best_count, time_limit, max_iters=1000000):
    """Random search with local improvement."""
    # Start from a grid
    current = _grid_config(3, 4)
    current_count = count_ordinary_lines(current)
    best_pts = [p[:] for p in current]
    best_count = current_count
    
    for _ in range(max_iters):
        if time.time() > time_limit:
            break
        # Random perturbation
        new_pts = [p[:] for p in current]
        i = random.randint(0, 11)
        new_pts[i][0] += random.randint(-50, 50)
        new_pts[i][1] += random.randint(-50, 50)
        
        if not _is_valid(new_pts):
            continue
        cnt = count_ordinary_lines(new_pts)
        if cnt <= current_count:
            current = new_pts
            current_count = cnt
            if cnt < best_count:
                best_count = cnt
                best_pts = [p[:] for p in current]
    
    return best_pts, best_count


def construct():
    """Find 12 integer points minimizing ordinary lines."""
    global _START_TIME, _DEADLINE
    
    # Initial best: 3x4 grid
    best_pts = _grid_config(3, 4)
    best_count = count_ordinary_lines(best_pts)
    
    time_limit = _DEADLINE - 5  # safety margin
    
    # Try structured configurations
    best_pts, best_count = _search_grid_configs(best_pts, best_count, time_limit)
    
    # Random search with local improvement
    best_pts, best_count = _random_search(best_pts, best_count, time_limit)
    
    # Ensure valid
    if not _is_valid(best_pts):
        best_pts = _grid_config(3, 4)
    
    return best_pts
# EVOLVE-BLOCK-END
