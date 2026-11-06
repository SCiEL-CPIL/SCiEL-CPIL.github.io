"""Maximize largest multiplicity of a single positive doubled area among 30 points."""
import time
import random
from itertools import combinations
from collections import Counter

_START_TIME = time.time()
_DEADLINE = 75.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _doubled_areas(points):
    """Compute all non-zero doubled areas for a list of points."""
    areas = []
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = xi*(yj - yk) - yi*(xj - xk) + xj*(yk - yj)
                d = abs(d)
                if d > 0:
                    areas.append(d)
    return areas

def _score(points):
    """Return (max_multiplicity, best_area) for a configuration."""
    areas = _doubled_areas(points)
    if not areas:
        return 0, 0
    c = Counter(areas)
    best_area, best_mult = c.most_common(1)[0]
    return best_mult, best_area

def _initial_grid():
    """Base grid: 6x5 points (seed)."""
    return [(x, y) for x in range(6) for y in range(5)]

def _perturb(points, rng, n_moves=3):
    """Randomly move some points to new positions."""
    pts = list(points)
    for _ in range(n_moves):
        idx = rng.randrange(30)
        x = rng.randint(-30, 30)
        y = rng.randint(-30, 30)
        # ensure distinct
        if (x, y) in pts and pts[idx] != (x, y):
            continue
        pts[idx] = (x, y)
    # deduplicate
    seen = set()
    unique = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    while len(unique) < 30:
        while True:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                unique.append((x, y))
                break
    return unique

def _try_lattice_construction(target_area, rng):
    """Try to build a configuration with many triples of target_area."""
    # Strategy: place points so that many triples have the same area.
    # For area A, a triangle with vertices (0,0), (a,0), (b,c) has area |a*c|/2,
    # so doubled area = |a*c|.
    # If we place points on a grid, many triples will have various areas.
    # Better: use a structured construction.
    
    # Construction: take points (i, 0) for i in range and (0, j) for j in range
    # and mix. Actually, let's try: place 15 points on x-axis and 15 on y-axis?
    # No - collinear points give 0 area for triples on same line.
    
    # Better: use a parallelogram-based construction.
    # Points: (i*a, j*b) for various i,j - grid gives area |a*b| for unit cells.
    # A 6x5 grid of step (a,b): points (i*a, j*b), i in [0,5], j in [0,4]
    # Doubled area of triangle (i1,j1),(i2,j2),(i3,j3) = |a*b| * |det of integer coords|
    # So all doubled areas are multiples of a*b.
    # The minimum non-zero doubled area is a*b (for unit triangles).
    # How many unit triangles in a 6x5 grid?
    # In a 6x5 grid of lattice points, the number of unit-area triangles...
    
    # Let's try different grid spacings
    best = None
    for a in range(1, 12):
        for b in range(1, 12):
            if a*b > 900:  # max doubled area for 30 points in [-30,30]
                continue
            # 6x5 grid with spacing (a,b)
            pts = [(i*a, j*b) for i in range(6) for j in range(5)]
            # Check bounds
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                mult, area = _score(pts)
                if best is None or mult > best[0]:
                    best = (mult, area, pts)
    return best

def _local_search(points, rng, max_iters=200):
    """Hill-climbing: try moving one point at a time."""
    pts = list(points)
    cur_mult, _ = _score(pts)
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        improved = False
        # Try moving each point
        for idx in range(30):
            if _elapsed() > _DEADLINE:
                break
            for _ in range(20):
                nx = rng.randint(-30, 30)
                ny = rng.randint(-30, 30)
                if (nx, ny) in pts:
                    continue
                old = pts[idx]
                pts[idx] = (nx, ny)
                new_mult, _ = _score(pts)
                if new_mult > cur_mult:
                    cur_mult = new_mult
                    improved = True
                    break
                else:
                    pts[idx] = old
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(3):
                if _elapsed() > _DEADLINE:
                    break
                new_pts = _perturb(pts, rng, n_moves=rng.randint(1, 5))
                new_mult, _ = _score(new_pts)
                if new_mult > cur_mult:
                    pts = new_pts
                    cur_mult = new_mult
                    improved = True
            if not improved:
                break
    return pts, cur_mult

def construct():
    rng = random.Random(42)
    best_pts = _initial_grid()
    best_mult, best_area = _score(best_pts)
    
    # Try lattice constructions
    lat_best = _try_lattice_construction(0, rng)
    if lat_best and lat_best[0] > best_mult:
        best_mult, best_area, best_pts = lat_best[0], lat_best[1], lat_best[2]
    
    # Local search from best
    if _elapsed() < _DEADLINE - 5:
        improved_pts, improved_mult = _local_search(best_pts, rng, max_iters=100)
        if improved_mult > best_mult:
            best_pts, best_mult = improved_pts, improved_mult
    
    # More local search iterations if time permits
    while _elapsed() < _DEADLINE - 10:
        improved_pts, improved_mult = _local_search(best_pts, rng, max_iters=50)
        if improved_mult > best_mult:
            best_pts, best_mult = improved_pts, improved_mult
        else:
            # Try random restarts
            new_pts = _perturb(best_pts, rng, n_moves=rng.randint(3, 8))
            new_mult, _ = _score(new_pts)
            if new_mult > best_mult:
                best_pts, best_mult = new_pts, new_mult
            improved_pts, improved_mult = _local_search(new_pts, rng, max_iters=30)
            if improved_mult > best_mult:
                best_pts, best_mult = improved_pts, improved_mult
    
    # Final: also try a few more structured constructions
    if _elapsed() < _DEADLINE - 5:
        # Try rectangular grids with different aspect ratios
        for cols in range(3, 11):
            for rows in range(3, 11):
                if cols * rows < 30 or cols * rows > 30:
                    continue
                # Can't do exactly 30 unless cols*rows=30
                pass
        # Try 5x6 and 6x5 grids with various offsets and spacings
        for a in range(1, 8):
            for b in range(1, 8):
                if a*b > 500:
                    continue
                for ox in range(-10, 11, 5):
                    for oy in range(-10, 11, 5):
                        pts = [(ox + i*a, oy + j*b) for i in range(6) for j in range(5)]
                        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                            mult, area = _score(pts)
                            if mult > best_mult:
                                best_pts, best_mult, best_area = pts, mult, area
    
    return best_pts
