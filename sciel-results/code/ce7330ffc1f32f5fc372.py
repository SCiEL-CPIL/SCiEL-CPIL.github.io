"""Deterministic algebraic construction + fast local refinement for 12-point few-ordinary-lines.

The core idea: instead of relying on stochastic search from random seeds (which tends to
get stuck in local minima with >6 ordinary lines), we use structured seeds derived from
known combinatorial geometries (grids, Hesse-like configurations, conic intersections)
and then apply a fast, deterministic local search that moves points along existing lines
to create more 3-point (or higher) collinearities.

Key structural change:
- Seeds are generated from algebraic/combinatorial constructions, not random points.
- Local search is biased toward "snap to line" moves that place a point on a line
  through two other points, which is the primary mechanism for reducing ordinary lines.
- The objective is computed incrementally where possible for speed.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 142.0  # safety margin before 150s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
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
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points (n=12)."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
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
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 3)


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


def _make_seed_grid_3x4():
    """3x4 grid."""
    return [[i, j] for i in range(3) for j in range(4)]


def _make_seed_grid_4x3():
    """4x3 grid."""
    return [[i, j] for i in range(4) for j in range(3)]


def _make_seed_hexagonal():
    """Hexagonal lattice with 12 points."""
    import math
    points = [[0, 0]]
    for angle in range(6):
        x = int(round(math.cos(angle * math.pi / 3) * 2))
        y = int(round(math.sin(angle * math.pi / 3) * 2))
        points.append([x, y])
    for angle in range(5):
        x = int(round(math.cos((angle + 0.5) * math.pi / 3) * 4))
        y = int(round(math.sin((angle + 0.5) * math.pi / 3) * 4))
        points.append([x, y])
    return points


def _make_seed_boroczky():
    """Böröczky-type configuration."""
    return [
        [0, 0],
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
        [2, 0], [0, 2], [-2, 0]
    ]


def _make_seed_conic():
    """Points on a conic (ellipse) plus structured additions."""
    # Use points on x^2 + y^2 = r^2 for various radii
    points = []
    # Points on unit circle (integer points)
    unit = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    points.extend([list(p) for p in unit])
    # Points on circle of radius 5: 3-4-5 triangle
    r5 = [(3, 4), (4, 3), (-3, 4), (-4, 3), (3, -4), (4, -3), (-3, -4), (-4, -3)]
    # Pick 8 more to make 12
    for p in r5:
        if len(points) >= 12:
            break
        points.append(list(p))
    return points[:12]


def _make_seed_hesse():
    """Hesse configuration-like: 9 inflection points of a cubic + 3 more."""
    # The 9 inflection points of x^3 + y^3 + z^3 - 3kxyz = 0 in affine coords
    # For k=1: points are (1,1), (1,ω), (1,ω^2), etc. But we need integer points.
    # Use a scaled version: 3x3 grid with some symmetry
    pts = [[i, j] for i in range(-1, 2) for j in range(-1, 2)]
    # Add 3 more points to break symmetry
    pts.append([2, 0])
    pts.append([0, 2])
    pts.append([-2, 0])
    return pts


def _egcd(a, b):
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _find_integer_point_on_line(p1, p2, t_range=(-20, 20)):
    """Find an integer point on the line through p1 and p2, parameterized by t."""
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
    # General solution: x = x0 + b*t, y = y0 - a*t
    # Pick t to get a point near the midpoint
    mid_t = 0
    # We want a point that's not p1 or p2
    # p1 corresponds to some t1, p2 to t2
    # Just pick a few t values
    candidates = []
    for t in range(t_range[0], t_range[1] + 1):
        nx = x0 + b * t
        ny = y0 - a * t
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            candidates.append((nx, ny))
    return candidates


def _local_search(points, max_time=10.0):
    """Deterministic local search biased toward snap-to-line moves."""
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        improved = False
        # Try all snap-to-line moves
        for i in range(12):
            if time.time() - start_time >= max_time:
                break
            others = [k for k in range(12) if k != i]
            # Try lines through pairs of other points
            for j_idx in range(len(others)):
                for k_idx in range(j_idx + 1, len(others)):
                    j = others[j_idx]
                    k = others[k_idx]
                    candidates = _find_integer_point_on_line(cur[j], cur[k], (-15, 15))
                    if not candidates:
                        continue
                    for (nx, ny) in candidates:
                        if (nx, ny) == tuple(cur[i]):
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = [nx, ny]
                        if not _valid(new_pts):
                            continue
                        new_obj = _objective(new_pts)
                        if new_obj < cur_obj:
                            cur = new_pts
                            cur_obj = new_obj
                            improved = True
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best = [p[:] for p in cur]
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            # Try small perturbations
            rng = random.Random(int(time.time() * 1000) % 2**32)
            for _ in range(20):
                if time.time() - start_time >= max_time:
                    break
                idx = rng.randrange(12)
                new_pts = [p[:] for p in cur]
                new_pts[idx][0] += rng.randint(-3, 3)
                new_pts[idx][1] += rng.randint(-3, 3)
                if not _valid(new_pts):
                    continue
                new_obj = _objective(new_pts)
                if new_obj < cur_obj:
                    cur = new_pts
                    cur_obj = new_obj
                    improved = True
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best = [p[:] for p in cur]
                    break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best, best_obj


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses structured seeds + local search."""
    
    best_pts = None
    best_score = float('inf')
    
    # Structured seeds
    seeds = [
        _make_seed_grid_3x4(),
        _make_seed_grid_4x3(),
        _make_seed_hexagonal(),
        _make_seed_boroczky(),
        _make_seed_conic(),
        _make_seed_hesse(),
    ]
    
    # Add a few random seeds for diversity
    rng = random.Random(42)
    for _ in range(4):
        pts = [[rng.randint(-50, 50), rng.randint(-50, 50)] for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
    
    # Run local search on each seed
    time_per_seed = 12.0
    
    for seed in seeds:
        if time.time() > _DEADLINE - 10:
            break
        if not _valid(seed):
            continue
        
        pts, score = _local_search(seed, max_time=time_per_seed)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
    
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 25:
        pts, score = _local_search(best_pts, max_time=25.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
    
    # Final local search refinement
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts, score = _local_search(best_pts, max_time=10.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        # Fallback
        best_pts = _make_seed_grid_3x4()
    
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_seed_grid_3x4()
    
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
