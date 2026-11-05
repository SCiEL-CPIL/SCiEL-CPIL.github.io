"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Uses Simulated Annealing to search for integer coordinates that minimize the number of
ordinary lines. The objective function penalizes ordinary lines and rewards 3-point lines.
"""
import time
import random
from math import gcd
import math

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


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


def _random_point():
    """Generate a random valid integer point."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]


def _make_seed_grid():
    """Create a 4x3 grid seed."""
    points = []
    for i in range(4):
        for j in range(3):
            points.append([i, j])
    return points


def _make_seed_hexagonal():
    """Create a hexagonal lattice seed with 12 points."""
    points = []
    points.append([0, 0])
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
    """Create a Böröczky-type configuration seed."""
    points = [
        [0, 0],
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
        [2, 0], [0, 2], [-2, 0]
    ]
    return points


def _simulate_annealing(seed_points, max_time=10.0):
    """Run Simulated Annealing to optimize a configuration."""
    rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        # Try to fix validity
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
        if not _valid(cur):
            return None, float('inf')
            
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 500.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        # Generate neighbor
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Small perturbation - move one point slightly
            idx = rng.randrange(12)
            delta_x = rng.randint(-3, 3)
            delta_y = rng.randint(-3, 3)
            new_pts[idx][0] += delta_x
            new_pts[idx][1] += delta_y
        elif move_type < 0.6:
            # Swap two points' positions
            i, j = rng.sample(range(12), 2)
            new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
        elif move_type < 0.8:
            # Move point to align with line through two other points
            i = rng.randrange(12)
            others = [k for k in range(12) if k != i]
            j, k = rng.sample(others, 2)
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
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
                continue
            if -C % g_ab != 0:
                continue
            # Find one integer solution using extended Euclidean
            a = A // g_ab
            b = B // g_ab
            c = -C // g_ab
            
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g_gcd, x0, y0 = egcd(a, b)
            if g_gcd != 1:
                continue
            x0 *= c
            y0 *= c
            # General solution: x = x0 + b*t, y = y0 - a*t
            t = rng.randint(-5, 5)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts[i] = [nx, ny]
        else:
            # Large jump - replace one point with random
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


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses Simulated Annealing."""
    
    best_pts = None
    best_score = float('inf')
    
    # Initial seeds - diverse configurations
    seeds = [
        _make_seed_grid(),
        _make_seed_hexagonal(),
        _make_seed_boroczky(),
    ]
    
    # Add random seeds
    for _ in range(5):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
            
    # Run SA for each seed
    time_per_seed = 12.0
    
    for seed in seeds:
        if time.time() > _DEADLINE - 10:
            break
            
        pts, score = _simulate_annealing(seed, max_time=time_per_seed)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 25:
        pts, score = _simulate_annealing(best_pts, max_time=25.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # Final local search refinement
    if best_pts is not None and time.time() < _DEADLINE - 5:
        cur = [p[:] for p in best_pts]
        cur_obj = _objective(cur)
        improved = True
        while improved and time.time() < _DEADLINE - 2:
            improved = False
            for i in range(12):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
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
                            improved = True
                            if cur_obj < best_score:
                                best_score = cur_obj
                                best_pts = [p[:] for p in cur]
    
    if best_pts is None:
        # Fallback
        best_pts = _make_seed_grid()
        
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_seed_grid()
        
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
