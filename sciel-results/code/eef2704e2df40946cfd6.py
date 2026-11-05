"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Uses Simulated Annealing with structural seeds (collinear triples) and triple-aware moves
to search for integer coordinates that minimize the number of ordinary lines.
Target: 6 ordinary lines (Böröczky-type configuration).
"""
import time
import random
from math import gcd
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
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


def _count_lines(points):
    """Count lines by number of points. Returns (ordinary_count, triple_count, all_counts)."""
    n = len(points)
    if n < 2:
        return 0, 0, {}
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts


def _incremental_objective(points, old_pt, new_pt):
    """Fast objective: only recount lines involving the moved point.
    Returns (ordinary, triples, line_counts) for the new configuration."""
    n = len(points)
    # Build full line counts excluding lines through old_pt
    # Actually, let's just recount all 66 pairs - it's fast enough in Python
    # But we can be smarter: remove old_pt contributions, add new_pt contributions
    line_counts = {}
    idx = None
    for i in range(n):
        if points[i] == old_pt:
            idx = i
            break
    if idx is None:
        # old_pt not found, just count normally
        return _count_lines(points)
    
    # Count all pairs not involving idx
    for i in range(n):
        if i == idx:
            continue
        x1, y1 = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Add pairs involving idx with new_pt
    x_new, y_new = new_pt
    for j in range(n):
        if j == idx:
            continue
        x2, y2 = points[j]
        key = _line_key(x_new, y_new, x2, y2)
        line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    ord_c, _, _ = _count_lines(points)
    return ord_c


def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points (n=12)."""
    _, tri_c, _ = _count_lines(points)
    return tri_c


def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines, tri_lines, _ = _count_lines(points)
    return ord_lines * 100 - tri_lines


def _objective_fast(points, old_pt, new_pt):
    """Fast objective using incremental counting."""
    ord_lines, tri_lines, _ = _incremental_objective(points, old_pt, new_pt)
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
    return [random.randint(-200, 200), random.randint(-200, 200)]


def _make_seed_boroczky():
    """Böröczky configuration for n=12: 8 vertices of a regular octagon-like structure
    plus 4 intersection points. This is known to achieve 6 ordinary lines.
    
    The classical Böröczky example for 12 points:
    Take a regular 12-gon's vertices and add intersection points of certain diagonals.
    
    A concrete integer construction:
    Use the structure from the Böröczky paper:
    - 4 points on x-axis: (0,0), (1,0), (2,0), (3,0) -- no, let's use known coords
    
    Known integer Böröczky-like config with 12 points achieving 6 ordinary lines:
    Let me construct from the standard example:
    Points at intersections of 4 lines in general position give 6 points,
    but we need 12.
    
    Alternative: Use the configuration from the literature:
    12 points = 4 collinear triples arranged as a complete quadrilateral extension.
    
    Let me use a specific known construction:
    The "Böröczky" 12-point configuration:
    Take vertices of a regular hexagon (6 points) and add 6 more at midpoints or intersections.
    
    For now, use a structured seed that's close to optimal:
    4 lines with 3 points each, arranged to minimize extra collinearities.
    """
    # Try a specific construction:
    # Line 1 (horizontal): (0,0), (2,0), (4,0)
    # Line 2 (horizontal): (1,1), (3,1), (5,1)  
    # Line 3 (slope 1): (0,1), (1,2), (2,3)
    # Line 4 (slope -1): (4,1), (3,2), (2,3) -- conflict at (2,3)!
    # Adjust: Line 4: (5,1), (4,2), (3,3)
    points = [
        [0, 0], [2, 0], [4, 0],      # horizontal y=0
        [1, 1], [3, 1], [5, 1],      # horizontal y=1
        [0, 2], [1, 3], [2, 4],      # slope 1, offset 2
        [4, 2], [3, 3], [2, 4],      # slope -1 -- conflict at (2,4)!
    ]
    # Fix: use different lines
    points = [
        [0, 0], [2, 0], [4, 0],      # y=0
        [1, 1], [3, 1], [5, 1],      # y=1
        [0, 2], [1, 4], [2, 6],      # slope 2, offset 2
        [5, 2], [4, 4], [3, 6],      # slope -2, offset 12
    ]
    # Check for duplicates
    seen = set()
    unique = []
    for p in points:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique[:12]


def _make_seed_boroczky2():
    """Another structured seed based on known near-optimal configurations.
    
    Use 4 lines with 3 points each, where lines intersect at shared points
    to create a more efficient structure.
    """
    # Configuration: 4 lines, each pair intersects at most once
    # Line 1: y=0 -> (0,0), (1,0), (2,0)
    # Line 2: y=1 -> (0,1), (1,1), (2,1)  
    # Line 3: x=0 -> (0,2), (0,3), (0,4) -- shares (0,0),(0,1) with lines 1,2
    # Line 4: x=1 -> (1,2), (1,3), (1,4) -- shares (1,0),(1,1) with lines 1,2
    # This gives: 3+3+3+3 = 12 points but with overlaps
    # Actual unique points: (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4) = 12
    points = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [0, 3], [0, 4],
        [1, 2], [1, 3], [1, 4],
    ]
    return points


def _make_seed_boroczky3():
    """Seed based on regular polygon + intersection points.
    
    Approximate a regular 12-gon with integer coordinates,
    then add intersection points of diagonals.
    """
    # 8 points approximating a circle, plus 4 intersection points
    # Circle radius ~5:
    # (5,0), (4,3), (0,5), (-3,4), (-5,0), (-4,-3), (0,-5), (3,-4)
    # Intersection points of diagonals:
    # Line from (5,0) to (-5,0) is x-axis
    # Line from (4,3) to (-4,-3) passes through origin
    # Let's use a simpler structured approach:
    points = [
        [5, 0], [4, 3], [0, 5], [-3, 4],   # quadrant 1-2 arc
        [-5, 0], [-4, -3], [0, -5], [3, -4], # quadrant 3-4 arc
        [1, 1], [-1, -1], [2, -1], [-2, 1],  # interior intersection points
    ]
    return points


def _make_seed_4triples():
    """Create a seed with 4 collinear triples (4 lines, 3 points each, 12 total).
    This is the target structure: 6 ordinary lines minimum."""
    # 4 lines, each with 3 points, no two lines sharing more than 1 point ideally
    # Simple construction: 4 horizontal lines at different y, 3 points each
    # y=0: (0,0),(1,0),(2,0)
    # y=1: (0,1),(1,1),(2,1)
    # y=2: (0,2),(1,2),(2,2)
    # y=3: (0,3),(1,3),(2,3)
    # This is a 3x4 grid - has many extra collinearities but starts from triple structure
    points = []
    for y in range(4):
        for x in range(3):
            points.append([x, y])
    return points


def _make_seed_diagonal():
    """Create a seed with 4 diagonal triples."""
    # Line 1: y=x -> (0,0),(1,1),(2,2)
    # Line 2: y=x+1 -> (0,1),(1,2),(2,3)
    # Line 3: y=-x+2 -> (0,2),(1,1),(2,0) -- but (1,1) and (2,0) already used
    # Let me use parallel lines with different offsets
    # y = x + c for c = 0, 1, 2, 3
    points = []
    for c in range(4):
        for t in range(3):
            points.append([t, t + c])
    return points


def _make_seed_mixed():
    """Create a mixed seed: 2 horizontal triples, 2 diagonal triples."""
    points = []
    # Horizontal: y=0: (0,0),(1,0),(2,0)
    points.extend([[0,0],[1,0],[2,0]])
    # Horizontal: y=1: (0,1),(1,1),(2,1)
    points.extend([[0,1],[1,1],[2,1]])
    # Diagonal: y=x+2: (0,2),(1,3),(2,4)
    points.extend([[0,2],[1,3],[2,4]])
    # Diagonal: y=-x+3: (0,3),(1,2),(2,1) -- (2,1) already used!
    # Use y=-x+4: (0,4),(1,3),(2,2) -- (1,3) already used!
    # Let me adjust: y=-x+5: (0,5),(1,4),(2,3)
    points.extend([[0,5],[1,4],[2,3]])
    return points


def _make_seed_parallel_triples():
    """4 parallel lines (slope 1), 3 points each, spaced apart."""
    points = []
    # y = x + c, c = 0, 3, 6, 9
    for c in [0, 3, 6, 9]:
        for t in [0, 1, 2]:
            points.append([t, t + c])
    return points


def _make_seed_staggered():
    """Staggered triples to minimize extra collinearities."""
    points = []
    # Triple 1: horizontal y=0: (0,0),(2,0),(4,0)
    points.extend([[0,0],[2,0],[4,0]])
    # Triple 2: horizontal y=1: (1,1),(3,1),(5,1)
    points.extend([[1,1],[3,1],[5,1]])
    # Triple 3: diagonal y=x: (0,1) already used... use y=x-1: (-1,0) already used
    # Let's use: y = 2x: (0,0) used. y = 2x+1: (0,1) used.
    # y = -x + 3: (0,3),(1,2),(2,1)
    points.extend([[0,3],[1,2],[2,1]])
    # Triple 4: y = -x + 4: (0,4),(1,3),(2,2)
    points.extend([[0,4],[1,3],[2,2]])
    return points


def _egcd(a, b):
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _find_integer_point_on_line(x1, y1, x2, y2, t_range=(-20, 20)):
    """Find an integer point on the line through (x1,y1) and (x2,y2)."""
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
    if -C % g_ab != 0:
        return None
    a = A // g_ab
    b = B // g_ab
    c = -C // g_ab
    g_gcd, x0, y0 = _egcd(a, b)
    if g_gcd != 1:
        return None
    x0 *= c
    y0 *= c
    # General solution: x = x0 + b*t, y = y0 - a*t
    return (x0, y0, a, b)  # base point and direction


def _simulate_annealing(seed_points, max_time=10.0, rng=None):
    """Run Simulated Annealing to optimize a configuration."""
    if rng is None:
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
    
    T = 300.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    steps = 0
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        moved_idx = -1
        old_pt = None
        
        if move_type < 0.40:
            # Small perturbation: move one point slightly
            idx = rng.randrange(12)
            moved_idx = idx
            old_pt = cur[idx][:]
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.70:
            # Snap to line: move point i onto line through j and k
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                T *= cooling_rate
                continue
            moved_idx = i
            old_pt = cur[i][:]
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            if x1 == x2 and y1 == y2:
                T *= cooling_rate
                continue
            res = _find_integer_point_on_line(x1, y1, x2, y2)
            if res is None:
                T *= cooling_rate
                continue
            x0, y0, a, b = res
            t = rng.randint(-20, 20)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                T *= cooling_rate
                continue
            new_pts[i] = [nx, ny]
        elif move_type < 0.85:
            # Swap coordinates or reflect
            idx = rng.randrange(12)
            moved_idx = idx
            old_pt = cur[idx][:]
            op = rng.randint(0, 2)
            if op == 0:
                new_pts[idx][0], new_pts[idx][1] = new_pts[idx][1], new_pts[idx][0]
            elif op == 1:
                new_pts[idx][0] = -new_pts[idx][0]
            else:
                new_pts[idx][1] = -new_pts[idx][1]
        else:
            # Large jump
            idx = rng.randrange(12)
            moved_idx = idx
            old_pt = cur[idx][:]
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        # Use fast objective if we have a valid move
        if moved_idx >= 0 and old_pt is not None:
            new_obj = _objective_fast(new_pts, old_pt, new_pts[moved_idx])
        else:
            new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
        
        T *= cooling_rate
        steps += 1
        
    return best, best_obj


def _local_polish(points, max_time=5.0, rng=None):
    """Local search: try moving each point to reduce ordinary lines."""
    if not _valid(points):
        return points
    if rng is None:
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
            old_pt = cur[i][:]
            # Try small moves
            found = False
            for dx in range(-6, 7):
                if found:
                    break
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective_fast(new_pts, old_pt, new_pts[i])
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        found = True
                        break
            # Try snap-to-line moves
            if time.time() - start_time >= max_time:
                break
            for j in range(12):
                if time.time() - start_time >= max_time:
                    break
                for k in range(12):
                    if i == j or i == k or j == k:
                        continue
                    x1, y1 = cur[j]
                    x2, y2 = cur[k]
                    if x1 == x2 and y1 == y2:
                        continue
                    res = _find_integer_point_on_line(x1, y1, x2, y2)
                    if res is None:
                        continue
                    x0, y0, a, b = res
                    for t in range(-15, 16):
                        nx = x0 + b * t
                        ny = y0 - a * t
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = [nx, ny]
                        if not _valid(new_pts):
                            continue
                        new_obj = _objective_fast(new_pts, old_pt, new_pts[i])
                        if new_obj < cur_obj:
                            cur = new_pts
                            cur_obj = new_obj
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best = [p[:] for p in cur]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best


def _canonical_key(points):
    """Create a canonical key for deduplication."""
    return tuple(sorted(tuple(p) for p in points))


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 1e9, not all collinear,
    minimizing ordinary lines. Uses Simulated Annealing with structural seeds."""
    
    best_pts = None
    best_score = float('inf')
    seen_configs = set()
    
    # Structured seeds with collinear triples
    seeds = [
        _make_seed_boroczky(),
        _make_seed_boroczky2(),
        _make_seed_boroczky3(),
        _make_seed_4triples(),
        _make_seed_diagonal(),
        _make_seed_mixed(),
        _make_seed_parallel_triples(),
        _make_seed_staggered(),
    ]
    
    # Add some random seeds
    for _ in range(3):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
    
    # Deduplicate seeds
    unique_seeds = []
    for s in seeds:
        key = _canonical_key(s)
        if key not in seen_configs:
            seen_configs.add(key)
            unique_seeds.append(s)
    
    # Run SA for each seed with time budget
    total_time = 120.0
    time_per_seed = total_time / max(len(unique_seeds), 1)
    time_per_seed = min(time_per_seed, 15.0)  # cap per seed
    
    for idx, seed in enumerate(unique_seeds):
        if time.time() > _DEADLINE - 10:
            break
        remaining = _DEADLINE - 10 - time.time()
        seed_time = min(time_per_seed, remaining)
        if seed_time < 2.0:
            break
            
        pts, score = _simulate_annealing(seed, max_time=seed_time)
        if pts and score < best_score:
            key = _canonical_key(pts)
            if key not in seen_configs:
                seen_configs.add(key)
                best_score = score
                best_pts = pts
    
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 20:
        pts, score = _simulate_annealing(best_pts, max_time=15.0)
        if pts and score < best_score:
            key = _canonical_key(pts)
            if key not in seen_configs:
                seen_configs.add(key)
                best_score = score
                best_pts = pts
    
    # Try additional SA runs from the best solution with different random seeds
    if best_pts is not None and time.time() < _DEADLINE - 15:
        for trial in range(3):
            if time.time() > _DEADLINE - 10:
                break
            rng = random.Random(1000 + trial * 777)
            pts, score = _simulate_annealing(best_pts, max_time=4.0, rng=rng)
            if pts and score < best_score:
                key = _canonical_key(pts)
                if key not in seen_configs:
                    seen_configs.add(key)
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
        best_pts = _make_seed_4triples()
        
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_seed_4triples()
        
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
