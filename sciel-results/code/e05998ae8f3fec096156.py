"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before the 150s kill


def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary


def _line_through(p1, p2):
    """Return canonical (a, b, c) for line through p1, p2: a*x+b*y=c."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dy), abs(dx))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    return (a, b, c)


def _points_on_line(pts, abc):
    """Count points on line a*x+b*y=c."""
    a, b, c = abc
    cnt = 0
    for (px, py) in pts:
        if a * px + b * py == c:
            cnt += 1
    return cnt


def _all_lines(pts):
    """Return dict: line_key -> list of point indices on that line."""
    n = len(pts)
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            abc = _line_through((x1, y1), (x2, y2))
            if abc is None:
                continue
            if abc not in line_map:
                line_map[abc] = []
            line_map[abc].append(i)
            line_map[abc].append(j)
    # Deduplicate indices in each line
    for abc in line_map:
        line_map[abc] = sorted(set(line_map[abc]))
    return line_map


def _count_ordinary_from_lines(pts):
    """Count ordinary lines given precomputed line map."""
    line_map = _all_lines(pts)
    ordinary = 0
    for abc, indices in line_map.items():
        if len(indices) == 2:
            ordinary += 1
    return ordinary


def _try_move(pts, idx, new_pt, line_map):
    """Try moving point idx to new_pt; return (new_count, new_pts) or None."""
    x, y = new_pt
    # Check if new_pt collides with existing points
    for i, (px, py) in enumerate(pts):
        if i != idx and px == x and py == y:
            return None
    new_pts = list(pts)
    new_pts[idx] = (x, y)
    cnt = _count_ordinary(new_pts)
    return (cnt, new_pts)


def _find_integer_points_on_line(abc, near_pt, max_dist=50, max_count=20):
    """Find integer points on line a*x+b*y=c within max_dist of near_pt."""
    a, b, c = abc
    x0, y0 = near_pt
    candidates = []
    # Parametrize: if gcd(a,b)=g, then a*x+b*y=c has solutions
    # x = x_p + (b/g)*t, y = y_p - (a/g)*t for integer t
    g = math.gcd(abs(a), abs(b))
    if g == 0 or c % g != 0:
        return candidates
    # Find one particular solution
    # a*x + b*y = c
    # Use extended gcd
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x, y = egcd(b, a % b)
        return (g, y, x - (a // b) * y)
    
    if a == 0:
        if b == 0:
            return candidates
        if c % b != 0:
            return candidates
        y_val = c // b
        for dx in range(-max_dist, max_dist + 1):
            candidates.append((x0 + dx, y_val))
        return candidates[:max_count]
    
    if b == 0:
        if c % a != 0:
            return candidates
        x_val = c // a
        for dy in range(-max_dist, max_dist + 1):
            candidates.append((x_val, y0 + dy))
        return candidates[:max_count]
    
    g, x_g, y_g = egcd(abs(a), abs(b))
    # Adjust signs
    if a < 0:
        x_g = -x_g
    if b < 0:
        y_g = -y_g
    c_div = c // g
    x_p = x_g * c_div
    y_p = y_g * c_div
    step_x = b // g
    step_y = -a // g
    
    # Find t range such that point is within max_dist of (x0, y0)
    # (x_p + step_x*t - x0)^2 + (y_p + step_y*t - y0)^2 <= max_dist^2
    # Solve for t bounds
    t_min = -10000
    t_max = 10000
    # Narrow down
    for t in range(t_min, t_max + 1):
        x = x_p + step_x * t
        y = y_p + step_y * t
        dx = x - x0
        dy = y - y0
        if dx * dx + dy * dy <= max_dist * max_dist:
            candidates.append((x, y))
            if len(candidates) >= max_count:
                break
    return candidates


def _local_search(pts, max_time=100.0):
    """Local search: try moving each point to various positions."""
    best_pts = list(pts)
    best_count = _count_ordinary(best_pts)
    rng = random.Random(42)
    
    deadline = time.time() + max_time
    stuck_count = 0
    no_improve_rounds = 0
    
    while time.time() < deadline:
        if no_improve_rounds > 50:
            break
        
        improved = False
        # Try moving each point
        for idx in range(len(best_pts)):
            if time.time() > deadline:
                break
            x0, y0 = best_pts[idx]
            
            # Generate candidate positions
            candidates = set()
            
            # Key strategy: find lines with 2+ other points and place on them
            others = [i for i in range(len(best_pts)) if i != idx]
            line_counts = {}
            for a_i in range(len(others)):
                for b_i in range(a_i + 1, len(others)):
                    p1 = best_pts[others[a_i]]
                    p2 = best_pts[others[b_i]]
                    abc = _line_through(p1, p2)
                    if abc is None:
                        continue
                    if abc not in line_counts:
                        line_counts[abc] = set()
                    line_counts[abc].add(others[a_i])
                    line_counts[abc].add(others[b_i])
            
            # Prioritize lines with more points
            sorted_lines = sorted(line_counts.items(), key=lambda x: -len(x[1]))
            for abc, pt_set in sorted_lines[:30]:  # Top 30 lines
                if len(pt_set) >= 2:
                    pts_on_line = _find_integer_points_on_line(abc, (x0, y0), max_dist=100, max_count=10)
                    for p in pts_on_line:
                        if p != (x0, y0):
                            candidates.add(p)
            
            # Also try random nearby positions
            for _ in range(30):
                dx = rng.randint(-20, 20)
                dy = rng.randint(-20, 20)
                if dx == 0 and dy == 0:
                    continue
                candidates.add((x0 + dx, y0 + dy))
            
            # Try each candidate
            for cand in candidates:
                if time.time() > deadline:
                    break
                result = _try_move(best_pts, idx, cand, None)
                if result is None:
                    continue
                new_count, new_pts = result
                if new_count < best_count:
                    best_count = new_count
                    best_pts = new_pts
                    improved = True
                    no_improve_rounds = 0
                    break
                elif new_count == best_count:
                    # Accept with some probability (sideways move)
                    if rng.random() < 0.15:
                        best_pts = new_pts
                        no_improve_rounds = 0
            
            if improved:
                break
        
        if not improved:
            no_improve_rounds += 1
    
    return best_pts, best_count


def _count_ordinary_fast(points):
    """Fast count of ordinary lines using line grouping."""
    n = len(points)
    if n < 2:
        return 0
    pts = [(p[0], p[1]) for p in points]
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            abc = _line_through((x1, y1), (x2, y2))
            if abc is None:
                continue
            if abc not in line_map:
                line_map[abc] = set()
            line_map[abc].add(i)
            line_map[abc].add(j)
    ordinary = 0
    for abc, indices in line_map.items():
        if len(indices) == 2:
            ordinary += 1
    return ordinary


def _try_move_fast(pts, idx, new_pt):
    """Try moving point idx to new_pt; return (new_count, new_pts) or None."""
    x, y = new_pt
    # Check if new_pt collides with existing points
    for i, (px, py) in enumerate(pts):
        if i != idx and px == x and py == y:
            return None
    new_pts = list(pts)
    new_pts[idx] = (x, y)
    cnt = _count_ordinary_fast(new_pts)
    return (cnt, new_pts)


def construct():
    # Start with 3 x 4 grid
    points = [(x, y) for x in range(4) for y in range(3)]
    
    # Run local search with time budget
    if time.time() < _DEADLINE:
        points, _ = _local_search(points, max_time=_DEADLINE - time.time() - 5.0)
    
    # Convert to list of lists
    return [[x, y] for x, y in points]
# EVOLVE-BLOCK-END
