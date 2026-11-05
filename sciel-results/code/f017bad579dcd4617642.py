"""Explicit construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Use finite field configurations (Hessian in GF(11), etc.) to find
    12-point sets with minimal ordinary lines.
  - Lift modular points to large integer coordinates preserving collinearity.
  - Local search to refine.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150 s kill


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


def _hessian_configuration(p):
    """Generate the Hessian configuration in GF(p): 9 inflection points of a
    cubic curve. The Hessian has 12 lines, each with 3 points, and 9 points.
    We need 12 points, so we'll extend this.
    
    For p=11, the Hessian configuration consists of points [x:y:z] where
    x^3 + y^3 + z^3 - 3kxyz = 0 for some k.
    
    A simpler approach: use the 12 points from a known low-ordinary-line
    configuration. Let's try the 4x3 grid and variations first, then try
    structured finite field approaches.
    """
    # The Hessian configuration has 9 points and 12 lines (each line has 3 pts)
    # For 12 points, we can try: 9 Hessian points + 3 more carefully placed
    
    # Actually, let's use a different known configuration.
    # The Böröczky example for n=12: 
    # 6 points on a conic, 6 points on a line (or similar)
    # But let's try systematic finite field search
    
    # For now, return the classic 3x4 grid as fallback
    return [[x, y] for x in range(4) for y in range(3)]


def _ff_search_small(p, n_points=12, time_limit=30):
    """Search in GF(p) for n_points with few ordinary lines, then lift."""
    import itertools
    
    start_time = time.time()
    best_ordinary = float('inf')
    best_points_mod = None
    
    # Generate all points in affine GF(p)^2
    all_pts = [(x, y) for x in range(p) for y in range(p)]
    
    # For small p, we can enumerate subsets
    if p <= 13 and len(all_pts) <= 169:
        # Try structured configurations
        # 1) Grids
        for rows in range(2, min(5, n_points + 1)):
            for cols in range(2, min(6, n_points + 1)):
                if rows * cols == n_points:
                    for ox in range(p):
                        for oy in range(p):
                            if time.time() - start_time > time_limit:
                                break
                            pts = [(ox + i, oy + j) for i in range(cols) for j in range(rows)]
                            # Count ordinary lines in GF(p)
                            ordinary = _count_ordinary_ff(pts, p)
                            if ordinary < best_ordinary:
                                best_ordinary = ordinary
                                best_points_mod = pts[:]
        
        # 2) Lines with shared points
        # Try: 4 lines each with 3-4 points, sharing intersections
        for _ in range(5000):
            if time.time() - start_time > time_limit:
                break
            # Generate random line arrangement
            pts = set()
            for _ in range(4):
                # Random line in GF(p)
                if random.random() < 0.3:
                    # Horizontal
                    y0 = random.randint(0, p - 1)
                    x0 = random.randint(0, p - 1)
                    for dx in range(3):
                        pts.add((x0 + dx) % p, y0)
                elif random.random() < 0.6:
                    # Vertical
                    x0 = random.randint(0, p - 1)
                    y0 = random.randint(0, p - 1)
                    for dy in range(3):
                        pts.add(x0, (y0 + dy) % p)
                else:
                    # Sloped
                    x0 = random.randint(0, p - 1)
                    y0 = random.randint(0, p - 1)
                    slope = random.randint(1, p - 1)
                    for dx in range(3):
                        pts.add((x0 + dx) % p, (y0 + slope * dx) % p)
            
            if len(pts) == n_points:
                pts_list = list(pts)
                ordinary = _count_ordinary_ff(pts_list, p)
                if ordinary < best_ordinary:
                    best_ordinary = ordinary
                    best_points_mod = pts_list[:]
    
    return best_points_mod, best_ordinary


def _count_ordinary_ff(points, p):
    """Count ordinary lines in GF(p) affine plane."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2) in GF(p)
            dx = (x2 - x1) % p
            dy = (y2 - y1) % p
            # Normalize: if dx == 0, vertical line x = x1
            # if dy == 0, horizontal line y = y1
            # else slope = dy/dx mod p
            if dx == 0:
                key = ('v', x1 % p)
            elif dy == 0:
                key = ('h', y1 % p)
            else:
                # slope = dy * dx^{-1} mod p
                inv_dx = pow(dx, p - 2, p)
                slope = (dy * inv_dx) % p
                # intercept: y - slope*x = c mod p
                c = (y1 - slope * x1) % p
                key = ('s', slope, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _lift_to_integer(points_mod, p, scale=1000000):
    """Lift points from GF(p)^2 to integer coordinates.
    
    To preserve collinearity exactly, we need to be careful.
    Simple lifting (x, y) -> (x*scale, y*scale) preserves collinearity
    since it's a uniform scaling. But it doesn't change the ordinary line count.
    
    Better: use a projective transformation or just scale.
    """
    return [[x * scale, y * scale] for x, y in points_mod]


def _local_search(points, max_iters=200000):
    """Hill-climb: replace one point to reduce ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(42)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        improved = False
        
        # Try placing on lines through existing point pairs
        for _ in range(30):
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i == j:
                continue
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            k = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            nx = x1 + k * dx
            ny = y1 + k * dy
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        
        if improved:
            continue
        
        # Small perturbation
        for _ in range(15):
            nx = old[0] + rng.randint(-3, 3)
            ny = old[1] + rng.randint(-3, 3)
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
    
    return cur, cur_score


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


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

    # Seed 1: 3x4 grid
    seed1 = _make_seed_grid(3, 4)
    _consider(seed1)

    # Try finite field search for better structures
    if time.time() < _DEADLINE - 60:
        for p in [11, 13]:
            if time.time() > _DEADLINE - 30:
                break
            pts_mod, ord_mod = _ff_search_small(p, 12, time_limit=20)
            if pts_mod is not None:
                # Lift to integer
                lifted = _lift_to_integer(pts_mod, p, scale=1000000)
                _consider(lifted)
                # Also try different scales
                for scale in [500000, 2000000]:
                    if time.time() > _DEADLINE - 20:
                        break
                    lifted2 = _lift_to_integer(pts_mod, p, scale=scale)
                    _consider(lifted2)

    # Local search from best so far
    if best_pts is not None and time.time() < _DEADLINE - 20:
        pts_ls, score_ls = _local_search(best_pts, max_iters=150000)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # Try more structured seeds
    # 4x3 grid
    seed2 = _make_seed_grid(4, 3)
    _consider(seed2)
    if time.time() < _DEADLINE - 15:
        pts_ls2, score_ls2 = _local_search(seed2, max_iters=100000)
        if score_ls2 < best_score:
            best_score = score_ls2
            best_pts = pts_ls2

    # 2x6 grid
    seed3 = _make_seed_grid(2, 6)
    _consider(seed3)
    if time.time() < _DEADLINE - 10:
        pts_ls3, score_ls3 = _local_search(seed3, max_iters=100000)
        if score_ls3 < best_score:
            best_score = score_ls3
            best_pts = pts_ls3

    # 6x2 grid
    seed4 = _make_seed_grid(6, 2)
    _consider(seed4)
    if time.time() < _DEADLINE - 8:
        pts_ls4, score_ls4 = _local_search(seed4, max_iters=100000)
        if score_ls4 < best_score:
            best_score = score_ls4
            best_pts = pts_ls4

    # Final polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts_final, score_final = _local_search(best_pts, max_iters=100000)
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
