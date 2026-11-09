"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math
from collections import Counter
from math import gcd

def has_five_collinear_fast(points):
    """Fast check: any 5 collinear? Uses slope counting per point."""
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
        for cnt in slopes.values():
            # cnt other points on this line + point i itself = cnt+1 total
            if cnt + 1 >= 5:
                return True
    return False

def count_4point_lines(points):
    """Count lines containing exactly 4 of the given points. Optimized with early exit."""
    n = len(points)
    if n < 4:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_r = dx // g
            dy_r = dy // g
            if dx_r < 0 or (dx_r == 0 and dy_r < 0):
                dx_r = -dx_r
                dy_r = -dy_r
            a = dy_r
            b = -dx_r
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_key = (a, b, c)
            line_counts[line_key] = line_counts.get(line_key, 0) + 1
    
    four_point = 0
    for cnt in line_counts.values():
        # k(k-1)/2 = cnt, solve for k
        # k = (1 + sqrt(1 + 8*cnt)) / 2
        disc = 1 + 8 * cnt
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 4:
                four_point += 1
    return four_point

def has_five_collinear(points):
    """Check if any 5 points are collinear. Delegates to fast version."""
    return has_five_collinear_fast(points)

def validate(points):
    """Check all constraints: 16 distinct points, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if has_five_collinear(points):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: 150s kill, return by 110s
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Starting configurations to try
    starts = []
    
    # 4x4 grid
    starts.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted/scaled
    starts.append([[x*2, y*2] for x in range(4) for y in range(4)])
    
    # 4x4 grid with one row perturbed
    base = [[x, y] for x in range(4) for y in range(4)]
    pert = [list(p) for p in base]
    pert[0] = [0, 1]
    pert[1] = [1, 2]
    pert[2] = [2, 0]
    pert[3] = [3, 3]
    starts.append(pert)
    
    # Fewer random configs to save time
    for _ in range(5):
        pts = []
        attempts = 0
        while len(pts) < 16 and attempts < 500:
            attempts += 1
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 16:
            starts.append(pts)
    
    # Local search function - streamlined for speed
    def local_search(points, max_iter=100):
        nonlocal best_points, best_score
        if time.time() > deadline:
            return
        pts = [list(p) for p in points]
        score = count_4point_lines(pts)
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in pts]
        
        iters = 0
        while iters < max_iter and time.time() < deadline:
            iters += 1
            found_improve = False
            for idx in range(16):
                if time.time() > deadline:
                    return
                cx, cy = pts[idx]
                # Try small moves only for speed
                for radius in [1, 2, 3]:
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            if [nx, ny] in pts:
                                continue
                            new_pts = [list(p) for p in pts]
                            new_pts[idx] = [nx, ny]
                            if has_five_collinear_fast(new_pts):
                                continue
                            new_score = count_4point_lines(new_pts)
                            if new_score > score:
                                score = new_score
                                pts = new_pts
                                found_improve = True
                                if score > best_score:
                                    best_score = score
                                    best_points = [list(p) for p in pts]
                                break
                        if found_improve:
                            break
                    if found_improve:
                        break
                if found_improve:
                    break
            if not found_improve:
                break
    
    # Try each starting configuration with time guard
    for start in starts:
        if time.time() > deadline:
            break
        if not validate(start):
            continue
        local_search(start)
    
    # If we found something, return it; otherwise return the 4x4 grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
