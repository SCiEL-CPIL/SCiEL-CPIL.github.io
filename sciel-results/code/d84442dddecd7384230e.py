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
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        for j in range(i+1, n):
            x2, y2 = xs[j], ys[j]
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

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_count = float('inf')
    
    def try_config(pts):
        nonlocal best_points, best_count
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return True
        return False
    
    # Structural approach: Use known optimal/near-optimal configurations
    # For 12 points, minimum ordinary lines is 6.
    # Try configurations based on highly symmetric arrangements.
    
    # Configuration A: 3x4 grid with specific offsets to create extra collinearities
    # Key insight: in a 3x4 grid, we already have many 3-point lines.
    # Try to add points that create additional collinearities.
    
    # Base: 3x4 grid
    base_3x4 = [(i, j) for i in range(3) for j in range(4)]
    if try_config(base_3x4):
        return best_points
    
    # Try 4x3 grid
    base_4x3 = [(i, j) for i in range(4) for j in range(3)]
    if try_config(base_4x3):
        return best_points
    
    # Configuration B: Two rows of 6 with interleaved positions
    for d1 in [1, 2, 3]:
        for d2 in [1, 2, 3]:
            for off in range(0, 3):
                if time.time() > deadline:
                    break
                row1 = [(i * d1, 0) for i in range(6)]
                row2 = [(i * d2 + off, 1) for i in range(6)]
                if try_config(row1 + row2):
                    return best_points
    
    # Configuration C: Three rows of 4 with staggered positions
    for s in [1, 2, 3]:
        if time.time() > deadline:
            break
        for shift in range(0, 3):
            row1 = [(i * s, 0) for i in range(4)]
            row2 = [(i * s + shift, 1) for i in range(4)]
            row3 = [(i * s + 2 * shift, 2) for i in range(4)]
            if try_config(row1 + row2 + row3):
                return best_points
    
    # Configuration D: Points on a cubic curve y = ax^3 + bx^2 + cx + d
    # Cubic curves can have many collinear triples
    for a in [1, -1]:
        for b in [0, 1, -1]:
            for c in [0, 1, -2, 2]:
                if time.time() > deadline:
                    break
                for d in [0, 1, -1]:
                    pts = []
                    for t in range(-3, 4):
                        y = a * t**3 + b * t**2 + c * t + d
                        pts.append((t, y))
                    if try_config(pts):
                        return best_points
    
    # Configuration E: Use a 2x6 grid but with non-uniform spacing to create diagonals
    for x_spacings in [[1,1,1,1,1,1], [1,1,1,2,1,1], [1,2,1,1,1,1], [2,1,1,1,1,1]]:
        if time.time() > deadline:
            break
        for y_spacings in [[1,1,1,1,1,1], [1,1,1,2,1,1]]:
            xs = [0]
            for s in x_spacings:
                xs.append(xs[-1] + s)
            xs = xs[:6]
            ys = [0]
            for s in y_spacings:
                ys.append(ys[-1] + s)
            ys = ys[:6]
            pts = [(xs[i], ys[i]) for i in range(6)] + [(xs[i], ys[i] + 1) for i in range(6)]
            if try_config(pts):
                return best_points
    
    # Configuration F: Hexagonal-like arrangement
    # 7 points in hexagonal pattern + 5 more
    hex_pts = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (1,2)]
    for dx in [-1, 0, 1, 2, 3]:
        for dy in [0, 1, 2, 3]:
            if time.time() > deadline:
                break
            extra = [(dx + i, dy + i) for i in range(5)]
            all_pts = hex_pts + extra
            if try_config(all_pts):
                return best_points
    
    # Configuration G: Use points from a finite field construction
    # GF(11) based: take 12 points from a conic over integers
    # y^2 = x^3 + ax + b with integer points
    for a in [0, 1, -1, 2]:
        for b in [0, 1, -1, 2, -2]:
            if time.time() > deadline:
                break
            pts = []
            for x in range(-5, 6):
                y2 = x**3 + a*x + b
                # Check if y2 is a perfect square
                if y2 >= 0:
                    y = int(y2**0.5)
                    if y * y == y2:
                        pts.append((x, y))
                        if y != 0:
                            pts.append((x, -y))
                if len(pts) >= 12:
                    pts = pts[:12]
                    break
            if len(pts) == 12:
                if try_config(pts):
                    return best_points
    
    # Configuration H: Deliberately construct with 4 collinear points
    # 4 on x-axis, 4 on y=1, 4 on y=2 with careful x-positions
    for x1 in range(-2, 3):
        for x2 in range(-2, 3):
            for x3 in range(-2, 3):
                if time.time() > deadline:
                    break
                for x4 in range(-2, 3):
                    row1 = [(0,0), (1,0), (2,0), (3,0)]
                    row2 = [(x1,1), (x1+1,1), (x1+2,1), (x1+3,1)]
                    row3 = [(x2,2), (x2+1,2), (x2+2,2), (x2+3,2)]
                    all_pts = row1 + row2 + row3
                    if try_config(all_pts):
                        return best_points
    
    # Configuration I: Use a 3x3 grid + 3 points at strategic locations
    grid3x3 = [(i, j) for i in range(3) for j in range(3)]
    for extra in [
        [(3,0), (0,3), (3,3)],
        [(3,1), (1,3), (3,3)],
        [(3,0), (0,3), (1,3)],
        [(3,0), (3,1), (0,3)],
        [(4,0), (0,4), (4,4)],
        [(3,0), (0,3), (4,4)],
        [(3,0), (4,3), (0,3)],
        [(3,0), (0,3), (3,4)],
    ]:
        if time.time() > deadline:
            break
        if try_config(grid3x3 + extra):
            return best_points
    
    # Configuration J: Use a 2x2x3 arrangement (two layers of 2x2 grids offset)
    for off_x in range(-2, 3):
        for off_y in range(-2, 3):
            if time.time() > deadline:
                break
            layer1 = [(i, j) for i in range(2) for j in range(3)]
            layer2 = [(i+2+off_x, j+2+off_y) for i in range(2) for j in range(3)]
            if try_config(layer1 + layer2):
                return best_points
    
    # If we haven't found 6, return best found
    if best_points is not None:
        return best_points
    
    # Fallback: return 3x4 grid
    return [[x, y] for x in range(3) for y in range(4)]
# EVOLVE-BLOCK-END
