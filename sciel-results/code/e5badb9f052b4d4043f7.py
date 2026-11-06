"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if points are valid: 12 distinct, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    for i in range(2, len(points)):
        x, y = points[i]
        if A * x + B * y + C != 0:
            return True
    return False

def _local_search(points, max_iters=5000):
    """Try to improve by moving individual points to reduce ordinary lines."""
    if not _is_valid(points):
        return points
    current_count = _count_ordinary(points)
    best_points = [list(p) for p in points]
    best_count = current_count
    
    for iteration in range(max_iters):
        improved = False
        for i in range(12):
            x, y = best_points[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check distinctness
                    new_pts = [list(p) for p in best_points]
                    new_pts[i] = [nx, ny]
                    if not _is_valid(new_pts):
                        continue
                    cnt = _count_ordinary(new_pts)
                    if cnt < best_count:
                        best_points = new_pts
                        best_count = cnt
                        improved = True
                        if best_count == 6:
                            return best_points
        if not improved:
            break
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_count = float('inf')
    
    # Configuration 1: Two parallel lines with 6 points each, carefully placed
    for offset in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        # Use arithmetic progression with common difference d
        for d in [1, 2, 3, 4, 5]:
            pts = [(i*d, 0) for i in range(6)] + [(i*d + offset//2, offset) for i in range(6)]
            if _is_valid(pts):
                cnt = _count_ordinary(pts)
                if cnt < best_count:
                    best_count = cnt
                    best_points = pts
                if best_count == 6:
                    return best_points
    
    # Configuration 2: Points on a conic (parabola) with symmetric placement
    for a in [1, 2, 3]:
        if time.time() > deadline:
            break
        for b in [0, 1, 2, 3]:
            pts = [(t, a*t*t + b) for t in range(-3, 4)]
            if _is_valid(pts):
                cnt = _count_ordinary(pts)
                if cnt < best_count:
                    best_count = cnt
                    best_points = pts
                if best_count == 6:
                    return best_points
    
    # Configuration 3: Grid with missing corners - 4x3 grid
    for xs, ys in [
        ([0,1,2,3], [0,1,2]),
        ([0,1,2,4], [0,1,2]),
        ([0,1,3,4], [0,1,2]),
        ([0,1,2,3], [0,1,3]),
        ([0,1,2,3], [0,2,3]),
        ([0,2,3,4], [0,1,2]),
        ([0,1,3,5], [0,1,2]),
        ([0,1,2,5], [0,1,2]),
    ]:
        if time.time() > deadline:
            break
        pts = [(x, y) for x in xs for y in ys]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
                if best_count == 6:
                    return best_points
    
    # Configuration 4: 3 points on each of 4 lines (forming a complete quadrilateral)
    # Lines: y=0, y=1, x=0, x=1 with intersections
    base_configs = [
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1),(4,0)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1),(3,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(4,0),(4,1),(4,2)],
    ]
    for pts in base_configs:
        if time.time() > deadline:
            break
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
                if best_count == 6:
                    return best_points
    
    # Configuration 5: Use local search from promising starting points
    if best_points is not None and time.time() < deadline - 30:
        improved = _local_search(best_points, max_iters=3000)
        if _is_valid(improved):
            cnt = _count_ordinary(improved)
            if cnt < best_count:
                best_count = cnt
                best_points = improved
            if best_count == 6:
                return best_points
    
    # Configuration 6: More systematic grid search
    if time.time() < deadline - 20:
        # Try 3x4 grids with various spacings
        for x_step in [1, 2, 3]:
            for y_step in [1, 2, 3]:
                if time.time() > deadline - 10:
                    break
                for x_off in range(0, 3):
                    for y_off in range(0, 3):
                        pts = [(x_off + i*x_step, y_off + j*y_step) 
                              for i in range(4) for j in range(3)]
                        if _is_valid(pts):
                            cnt = _count_ordinary(pts)
                            if cnt < best_count:
                                best_count = cnt
                                best_points = pts
                                if best_count == 6:
                                    return best_points
    
    # Configuration 7: Try configurations with 4 points on a line
    if time.time() < deadline - 15:
        for n_on_line in [4, 5, 6]:
            if time.time() > deadline - 10:
                break
            # n points on x-axis, remaining 12-n placed to minimize ordinary lines
            for n in [n_on_line]:
                pts = [(i, 0) for i in range(n)]
                # Place remaining points
                remaining = 12 - n
                # Try placing them in a row above
                for offset in [1, 2, 3]:
                    for d in [1, 2, 3]:
                        extra = [(i*d, offset) for i in range(remaining)]
                        all_pts = pts + extra
                        if _is_valid(all_pts):
                            cnt = _count_ordinary(all_pts)
                            if cnt < best_count:
                                best_count = cnt
                                best_points = all_pts
                                if best_count == 6:
                                    return best_points
    
    if best_points is not None:
        return best_points
    
    # Fallback
    return [[x, y] for x in range(4) for y in range(3)]
# EVOLVE-BLOCK-END
