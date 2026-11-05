"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Uses Simulated Annealing with targeted line-snapping moves to find integer coordinates
that minimize the number of ordinary lines. Key insight: the optimal configuration
has 6 points on a line, so we seed and target that structure.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points (n=12)."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 3)


def _get_line_counts(points):
    """Get dict of line -> count of points on that line."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return line_counts


def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines = _count_ordinary(points)
    tri_lines = _count_3point_lines(points)
    return ord_lines * 100 - tri_lines


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _random_point():
    """Generate a random valid integer point."""
    return [random.randint(-500, 500), random.randint(-500, 500)]


def _make_seed_boroczky():
    """Create a Böröczky-type seed: 6 on a line, others arranged for 3-point lines.
    
    Known optimal structure for 12 points: 6 collinear points, and the other 6
    arranged so that many 3-point lines exist.
    """
    # 6 points on x-axis
    pts = [[i, 0] for i in range(6)]
    # Add points to create 3-point lines through pairs of axis points
    # Point (a,b) creates 3-point lines with pairs on axis if it lies on lines through those pairs
    # For points (i,0) and (j,0), the line is y=0 (the axis itself)
    # We need points that create 3-point lines with 2 axis points + 1 other point
    # Or 3 non-axis points on a line
    # 
    # Strategy: place points at (0,1), (1,1), (2,1) -> line y=1 has 3 points
    # Place points at (0,2), (1,2), (2,2) -> line y=2 has 3 points
    # But we need to check if this creates enough 3-point lines
    pts.append([0, 1])
    pts.append([1, 1])
    pts.append([2, 1])
    pts.append([0, 2])
    pts.append([1, 2])
    pts.append([2, 2])
    return pts


def _make_seed_optimal_structure():
    """Create a seed based on known near-optimal 12-point configuration.
    
    Based on Böröczky's construction: 6 points on a line L, and for each pair
    of points on L, there's a point off L such that the line through that point
    and the pair creates additional 3-point lines.
    """
    # 6 points on x-axis: (0,0), (1,0), (2,0), (3,0), (4,0), (5,0)
    pts = [[i, 0] for i in range(6)]
    # Add 6 more points strategically
    # Points that create 3-point lines with axis points
    # (0,1): line through (0,1) and (1,0) has slope -1, check if any other point is on it
    # Let's use a known good configuration:
    # The Böröczky configuration for n=12 has 6 on a line and 6 off
    # Off-line points are chosen so that lines through pairs of off-line points
    # pass through axis points
    #
    # Try: (0,1), (2,1), (4,1), (1,2), (3,2), (5,2)
    pts.append([0, 1])
    pts.append([2, 1])
    pts.append([4, 1])
    pts.append([1, 2])
    pts.append([3, 2])
    pts.append([5, 2])
    return pts


def _make_seed_grid():
    """Create a 4x3 grid seed."""
    points = []
    for i in range(4):
        for j in range(3):
            points.append([i, j])
    return points


def _egcd(a, b):
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _find_integer_point_on_line(p1, p2, rng, t_range=(-15, 15)):
    """Find a random integer point on the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    # A x + B y = -C
    g_ab = gcd(abs(A), abs(B))
    if g_ab == 0:
        return None
    if (-C) % g_ab != 0:
        return None
    a = A // g_ab
    b = B // g_ab
    c = -C // g_ab
    g_gcd, x0, y0 = _egcd(a, b)
    if g_gcd != 1:
        return None
    x0 *= c
    y0 *= c
    t = rng.randint(t_range[0], t_range[1])
    nx = x0 + b * t
    ny = y0 - a * t
    if abs(nx) > 10**9 or abs(ny) > 10**9:
        return None
    return [nx, ny]


def _simulate_annealing(seed_points, max_time=10.0):
    """Run Simulated Annealing to optimize a configuration with targeted moves."""
    rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
        if not _valid(cur):
            return None, float('inf')
            
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 500.0
    T_min = 0.05
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.35:
            # Small perturbation: move one point slightly
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-3, 3)
            new_pts[idx][1] += rng.randint(-3, 3)
        elif move_type < 0.65:
            # Snap to a line through two other points (prefer lines with 2+ existing points)
            i = rng.randrange(12)
            # Get line counts to prefer busy lines
            line_counts = _get_line_counts(cur)
            # Find lines with count >= 2 (excluding lines through point i)
            candidate_lines = []
            for key, cnt in line_counts.items():
                if cnt >= 2:
                    # Check if point i is already on this line
                    A, B, C = key
                    if A * cur[i][0] + B * cur[i][1] + C == 0:
                        continue
                    candidate_lines.append((key, cnt))
            
            if candidate_lines:
                # Weight by count - prefer lines with more points
                weights = [cnt for _, cnt in candidate_lines]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen_key = candidate_lines[0][0]
                for key, cnt in candidate_lines:
                    cumulative += cnt
                    if r <= cumulative:
                        chosen_key = key
                        break
                # Find two points on this line to use for generating a new point
                A, B, C = chosen_key
                # Find two points on this line
                pts_on_line = []
                for idx2, pt in enumerate(cur):
                    if A * pt[0] + B * pt[1] + C == 0:
                        pts_on_line.append(idx2)
                if len(pts_on_line) >= 2:
                    p1_idx = pts_on_line[0]
                    p2_idx = pts_on_line[1]
                    new_pt = _find_integer_point_on_line(cur[p1_idx], cur[p2_idx], rng)
                    if new_pt is not None:
                        new_pts[i] = new_pt
                    else:
                        # Fallback: small perturbation
                        new_pts[i][0] += rng.randint(-3, 3)
                        new_pts[i][1] += rng.randint(-3, 3)
                else:
                    new_pts[i][0] += rng.randint(-3, 3)
                    new_pts[i][1] += rng.randint(-3, 3)
            else:
                # No busy lines, just perturb
                new_pts[i][0] += rng.randint(-3, 3)
                new_pts[i][1] += rng.randint(-3, 3)
        elif move_type < 0.85:
            # Snap to a line through two specific points (targeted)
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                T *= cooling_rate
                continue
            new_pt = _find_integer_point_on_line(cur[j], cur[k], rng)
            if new_pt is not None:
                new_pts[i] = new_pt
            else:
                new_pts[i][0] += rng.randint(-3, 3)
                new_pts[i][1] += rng.randint(-3, 3)
        else:
            # Large jump: replace a point with a random one
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
        T *= cooling_rate
        
    return best, best_obj


def _local_polish(points, max_time=5.0):
    """Local search: try moving each point to reduce ordinary lines."""
    if not _valid(points):
        return points
    rng = random.Random(42)
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            # Try small moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective(new_pts)
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        break
                else:
                    continue
                break
            # Try snapping to lines
            if time.time() - start_time >= max_time:
                break
            line_counts = _get_line_counts(cur)
            for key, cnt in line_counts.items():
                if cnt < 2:
                    continue
                A, B, C = key
                if A * cur[i][0] + B * cur[i][1] + C == 0:
                    continue
                # Find two points on this line
                pts_on_line = []
                for idx2, pt in enumerate(cur):
                    if A * pt[0] + B * pt[1] + C == 0:
                        pts_on_line.append(idx2)
                if len(pts_on_line) >= 2:
                    new_pt = _find_integer_point_on_line(cur[pts_on_line[0]], cur[pts_on_line[1]], rng, (-8, 8))
                    if new_pt is not None:
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = new_pt
                        if _valid(new_pts):
                            new_obj = _objective(new_pts)
                            if new_obj < cur_obj:
                                cur = new_pts
                                cur_obj = new_obj
                                if cur_obj < best_obj:
                                    best_obj = cur_obj
                                    best = [p[:] for p in cur]
                                improved = True
                                break
    return best


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses Simulated Annealing with targeted line-snapping."""
    
    best_pts = None
    best_score = float('inf')
    
    # Initial seeds - prioritize structures close to known optimal
    seeds = [
        _make_seed_optimal_structure(),
        _make_seed_boroczky(),
        _make_seed_grid(),
    ]
    
    # Add a few random seeds
    for _ in range(2):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
    
    # Allocate time: more time for better seeds
    # First seed gets more time as it's most promising
    time_budget = 140.0  # total time available
    elapsed = time.time() - _START
    
    # Run SA for each seed with decreasing time allocation
    for idx, seed in enumerate(seeds):
        if time.time() > _DEADLINE - 10:
            break
        # Allocate more time to earlier (better) seeds
        if idx == 0:
            t_alloc = min(45.0, time_budget - (time.time() - _START))
        elif idx == 1:
            t_alloc = min(30.0, time_budget - (time.time() - _START))
        elif idx == 2:
            t_alloc = min(20.0, time_budget - (time.time() - _START))
        else:
            t_alloc = min(10.0, time_budget - (time.time() - _START))
        
        if t_alloc < 5.0:
            break
            
        pts, score = _simulate_annealing(seed, max_time=t_alloc)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # If we found a good solution, try to refine it further with more SA
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 25:
        # Run SA again on the best configuration with a different seed
        rng_seed = int(time.time() * 1000) % 2**32
        pts, score = _simulate_annealing(best_pts, max_time=20.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # Final local polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        polished = _local_polish(best_pts, max_time=5.0)
        if polished:
            polished_obj = _objective(polished)
            if polished_obj < best_score:
                best_pts = polished
                best_score = polished_obj
    
    if best_pts is None:
        # Fallback
        best_pts = _make_seed_optimal_structure()
        
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_seed_optimal_structure()
        
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("3-point lines:", _count_3point_lines(pts))
    print("Objective:", _objective(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
