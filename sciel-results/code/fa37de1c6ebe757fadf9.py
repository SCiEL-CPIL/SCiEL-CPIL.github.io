import time
import math
from itertools import combinations

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, _gcd(dx, c))
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary_fast(points):
    n = len(points)
    if n < 3:
        return 0
    # For each pair, determine the line; count how many points are on each line.
    # A line is "ordinary" if exactly 2 points lie on it.
    # Strategy: build dict line_key -> count of points on that line.
    # But we must avoid double counting: each pair defines a line, and if k points
    # are on that line, it contributes C(k,2) pairs. We need to count lines with k==2.
    # Efficient approach: for each point i, for each j > i, compute line key, 
    # and count how many times each line key appears across all pairs.
    # Then for each line key that appears m times (as a pair), the number of points
    # on that line is k where C(k,2) = m, i.e., k = (1 + sqrt(1+8m))/2.
    # But this only works if all pairs on the line are counted. Actually, if k points
    # are on a line, there are C(k,2) pairs that define that line. So if we count
    # how many pairs map to each line key, we get C(k,2) for each line.
    # Then k = (1 + sqrt(1+8*m))/2, and the line is ordinary iff k == 2 (i.e., m == 1).
    
    pair_line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            pair_line_counts[key] = pair_line_counts.get(key, 0) + 1
    
    # A line is ordinary iff exactly one pair maps to it (i.e., exactly 2 points on it)
    return sum(1 for cnt in pair_line_counts.values() if cnt == 1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = float('inf')
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    def check_time():
        return time.time() > deadline
    
    # Known good configurations to seed the search
    
    # 3x4 grid
    grid34 = [[x, y] for x in range(3) for y in range(4)]
    s = _count_ordinary_fast(grid34)
    update_best(grid34, s)
    if best_score <= 6:
        return best_points
    
    # 4x3 grid
    grid43 = [[x, y] for x in range(4) for y in range(3)]
    s = _count_ordinary_fast(grid43)
    update_best(grid43, s)
    if best_score <= 6:
        return best_points
    
    # Try sheared grids
    for base, dims in [(((x, y) for x in range(3) for y in range(4)), (3, 4)),
                       (((x, y) for x in range(4) for y in range(3)), (4, 3))]:
        nx, ny = dims
        for k in range(-8, 9):
            if check_time():
                break
            pts = [[x + k * y, y] for x, y in base]
            if len(set(map(tuple, pts))) != 12:
                continue
            s = _count_ordinary_fast(pts)
            update_best(pts, s)
            if best_score <= 6:
                return best_points
    
    # Try more structured configurations
    
    # Configuration: 3 points on each of 4 lines (12 points total)
    # Lines: y=0, y=1, y=2, y=3, with points spaced to create extra collinearities
    # Try: (0,0),(1,0),(2,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (0,3),(1,3),(2,3)
    # This is the 3x4 grid already.
    
    # Try a "star" configuration: points on 3 lines through a common point
    # Line 1: y=0: (1,0),(2,0),(3,0),(4,0)
    # Line 2: y=x: (1,1),(2,2),(3,3),(4,4)
    # Line 3: y=-x: (1,-1),(2,-2),(3,-3),(4,-4)
    star = [[1,0],[2,0],[3,0],[4,0],[1,1],[2,2],[3,3],[4,4],[1,-1],[2,-2],[3,-3],[4,-4]]
    s = _count_ordinary_fast(star)
    update_best(star, s)
    if best_score <= 6:
        return best_points
    
    # Try: points forming a hexagonal lattice pattern
    # 12 points from a triangular lattice
    tri_pts = []
    # Center at (0,0), add neighbors in hexagonal pattern
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            # Triangular lattice: use basis vectors (1,0) and (0.5, sqrt(3)/2)
            # But we need integer coordinates. Let's use a different approach.
            # Use the fact that in a triangular lattice, points satisfy
            # certain linear relations.
            pass
    
    # Alternative: use a known configuration from the ordinary lines literature
    # For n=12, the minimum number of ordinary lines over the reals is 6.
    # One known configuration achieving 6 is the 3x4 grid with one point moved
    # to create additional collinearities.
    
    # Let's try a systematic search over small coordinate ranges
    # Use local search with pair swaps to find better configurations
    
    if best_points is not None:
        current = [p[:] for p in best_points]
        current_score = best_score
        
        # Generate a candidate pool
        pool = set()
        for x in range(-6, 7):
            for y in range(-6, 7):
                pool.add((x, y))
        pool = list(pool)
        
        max_rounds = 50
        for rnd in range(max_rounds):
            if check_time():
                break
            improved = False
            
            # Try single-point moves
            for i in range(12):
                if check_time():
                    break
                old_pt = current[i][:]
                for px, py in pool:
                    if (px, py) == tuple(old_pt):
                        continue
                    if any(tuple(current[j]) == (px, py) for j in range(12)):
                        continue
                    current[i] = [px, py]
                    new_score = _count_ordinary_fast(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        update_best(current, new_score)
                        if best_score <= 6:
                            return best_points
                        break
                    current[i] = old_pt[:]
                
                if best_score <= 6:
                    return best_points
            
            if not improved:
                break
            
            # Try pair swaps
            if check_time():
                break
            for i in range(12):
                if check_time():
                    break
                for j in range(i + 1, 12):
                    if check_time():
                        break
                    old_i = current[i][:]
                    old_j = current[j][:]
                    for px, py in pool:
                        if check_time():
                            break
                        if (px, py) == tuple(old_i) or (px, py) == tuple(old_j):
                            continue
                        if any(tuple(current[k]) == (px, py) for k in range(12)):
                            continue
                        current[i] = [px, py]
                        
                        # Find best replacement for j
                        best_swap_score = current_score
                        best_swap_pt = None
                        for qx, qy in pool:
                            if (qx, qy) == tuple(old_i) or (qx, qy) == (px, py):
                                continue
                            if any(tuple(current[k]) == (qx, qy) for k in range(12)):
                                continue
                            current[j] = [qx, qy]
                            s = _count_ordinary_fast(current)
                            if s < best_swap_score:
                                best_swap_score = s
                                best_swap_pt = [qx, qy]
                        
                        if best_swap_pt is not None:
                            current[j] = best_swap_pt
                            current_score = best_swap_score
                            improved = True
                            update_best(current, current_score)
                            if best_score <= 6:
                                return best_points
                        else:
                            current[i] = old_i[:]
                            current[j] = old_j[:]
    
    # If we still haven't found a good configuration, return the best we have
    if best_points is None:
        best_points = [[x, y] for x in range(3) for y in range(4)]
    
    return best_points
