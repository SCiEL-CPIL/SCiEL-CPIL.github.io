import time
import random
from math import gcd
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 120 s kill


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


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


def _make_seed_triangle():
    """
    Seed: 3 lines of 4 points, intersecting at 3 vertices.
    Line 1: y=0, x in {0, 1, 2, 3} -> (0,0), (1,0), (2,0), (3,0)
    Line 2: x=3, y in {0, 1, 2, 3} -> (3,0), (3,1), (3,2), (3,3)
    Line 3: x+y=3, points (0,3), (1,2), (2,1), (3,0)
    Union:
    (0,0), (1,0), (2,0), (3,0)
    (3,1), (3,2), (3,3)
    (0,3), (1,2), (2,1)
    Total 10 points. Need 2 more.
    Let's add (0,1) and (1,1)?
    Or extend the lines?
    Let's try a different structure: 4 lines of 3 points?
    """
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [3, 1], [3, 2], [3, 3],
        [0, 3], [1, 2], [2, 1],
        [0, 1], [1, 1]
    ]
    return pts

def _make_seed_hesse_extension():
    """
    Based on 3 lines of 4 points forming a triangle.
    Points:
    L1 (y=0): (0,0), (1,0), (2,0), (3,0)
    L2 (x=0): (0,0), (0,1), (0,2), (0,3)
    L3 (x+y=3): (0,3), (1,2), (2,1), (3,0)
    Union:
    (0,0), (1,0), (2,0), (3,0)
    (0,1), (0,2)
    (1,2), (2,1)
    Total 9 points.
    Add 3 more to make 12.
    Try adding (1,1), (2,2), (1,3)?
    (1,1) is on x=y.
    (2,2) is on x=y.
    (1,3) is on x=1.
    Let's just return a 4x3 grid as a robust seed.
    """
    return _make_seed_grid(3, 4)

def _simulated_annealing(points, max_iters=500000):
    """
    Simulated Annealing to minimize ordinary lines.
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    rng = random.Random(42)
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    iters = 0
    no_improve = 0
    
    while iters < max_iters and time.time() < _DEADLINE - 5 and T > min_T:
        iters += 1
        T *= cooling
        
        idx = rng.randrange(12)
        old_pt = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        
        # Generate candidate
        # Strategy: Move point to a random position in a small neighborhood
        # Or onto a line defined by two other points
        if rng.random() < 0.5:
            # Perturbation
            nx = old_pt[0] + rng.randint(-5, 5)
            ny = old_pt[1] + rng.randint(-5, 5)
            cand = (nx, ny)
        else:
            # Align with existing line
            j = rng.randrange(12)
            k = rng.randrange(12)
            if j == idx or k == idx or j == k:
                cand = (old_pt[0] + rng.randint(-5, 5), old_pt[1] + rng.randint(-5, 5))
            else:
                p1 = cur[j]
                p2 = cur[k]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
                if g:
                    dx //= g
                    dy //= g
                # Random step along the line
                m = rng.randint(-10, 10)
                cand = (p1[0] + m * dx, p1[1] + m * dy)
                
        if cand in cur_set:
            continue
        if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
            continue
            
        new_pts = [p[:] for p in cur]
        new_pts[idx] = list(cand)
        
        if not _valid(new_pts):
            continue
            
        new_score = _count_ordinary(new_pts)
        
        delta = new_score - cur_score
        if delta < 0 or rng.random() < (2.718281828 ** (-delta / T)):
            cur = new_pts
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        if no_improve > 10000:
            # Restart from best or perturb best
            cur = [p[:] for p in best]
            cur_score = best_score
            no_improve = 0
            # Optional: perturb best to escape
            if rng.random() < 0.5:
                idx = rng.randrange(12)
                cur[idx] = [cur[idx][0] + rng.randint(-10, 10), cur[idx][1] + rng.randint(-10, 10)]
                if _valid(cur):
                    cur_score = _count_ordinary(cur)
                else:
                    cur = [p[:] for p in best]
                    cur_score = best_score

    return best, best_score

def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Seed 1: 3x4 Grid
    seed1 = _make_seed_grid(3, 4)
    _consider(seed1)
    
    # Seed 2: Triangle-like structure
    seed2 = _make_seed_triangle()
    _consider(seed2)
    
    # Seed 3: Random points in small box
    rng_init = random.Random(99)
    seed3 = []
    seen = set()
    while len(seed3) < 12:
        x = rng_init.randint(-5, 5)
        y = rng_init.randint(-5, 5)
        if (x, y) not in seen:
            seen.add((x, y))
            seed3.append([x, y])
    _consider(seed3)
    
    # Seed 4: 4x3 Grid (transposed)
    seed4 = _make_seed_grid(4, 3)
    _consider(seed4)

    # Run Simulated Annealing from the best seed found so far
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_sa, score_sa = _simulated_annealing(best_pts, max_iters=200000)
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Run SA from other seeds if time permits
    if time.time() < _DEADLINE - 10:
        for seed in [seed1, seed2, seed3, seed4]:
            if _valid(seed):
                s = _count_ordinary(seed)
                if s <= best_score + 2:
                    pts_sa, score_sa = _simulated_annealing(seed, max_iters=100000)
                    if score_sa < best_score:
                        best_score = score_sa
                        best_pts = pts_sa
                        if best_score == 6:
                            break

    # Final polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts_final, score_final = _simulated_annealing(best_pts, max_iters=200000)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
