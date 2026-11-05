"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line (as a normalized (A,B,C) from Ax+By+C=0) to the set of points on it
    line_to_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            # Line through (xi,yi) and (xj,yj): A*x + B*y + C = 0
            # A = yj - yi, B = xi - xj, C = -(A*xi + B*yi)
            A = yj - yi
            B = xi - xj
            C = -(A * xi + B * yi)
            # Normalize: divide by gcd of A, B, C (handle signs)
            # Find gcd
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            # Normalize sign: make first nonzero positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    # Count lines with exactly 2 points
    ordinary = 0
    for pts_on_line in line_to_pts.values():
        if len(pts_on_line) == 2:
            ordinary += 1
    return ordinary

def _try_move(points, idx, candidate, current_score):
    """Try placing candidate at position idx, return new score or None if invalid."""
    new_points = points[:idx] + [candidate] + points[idx+1:]
    return _count_ordinary_lines(new_points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_score = None
    best_points = None
    
    # Generate initial diverse configurations
    random.seed(42)
    
    # Strategy 1: Start from 3x4 grid and optimize
    def optimize_from(start_points, max_iters=50000):
        nonlocal best_score, best_points, deadline
        current = start_points[:]
        current_score = _count_ordinary_lines(current)
        if best_score is None or current_score < best_score:
            best_score = current_score
            best_points = current[:]
        
        temp = 2.0
        cooling = 0.9995
        n = len(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if temp < 0.01:
                break
            
            # Pick a random point to move
            idx = random.randint(0, n - 1)
            old_pt = current[idx]
            
            # Try to find a better position
            found_improvement = False
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                # Generate candidate near other points to create collinearities
                other_idx = random.choice([i for i in range(n) if i != idx])
                other_pt = current[other_idx]
                if random.random() < 0.5:
                    # Place on line through two other points
                    other2_idx = random.choice([i for i in range(n) if i != idx and i != other_idx])
                    p2 = current[other2_idx]
                    dx = p2[0] - other_pt[0]
                    dy = p2[1] - other_pt[1]
                    t = random.randint(-3, 3)
                    cand = [other_pt[0] + t * dx, other_pt[1] + t * dy]
                else:
                    # Random small perturbation
                    cand = [other_pt[0] + random.randint(-5, 5), other_pt[1] + random.randint(-5, 5)]
                
                if cand == old_pt:
                    continue
                if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
                    continue
                # Check distinctness
                if any(c == p for c in cand for p in []):  # simplified
                    pass
                valid = True
                for i, p in enumerate(current):
                    if i != idx and p == cand:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_score = _count_ordinary_lines(current[:idx] + [cand] + current[idx+1:])
                delta = new_score - current_score
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    current[idx] = cand
                    current_score = new_score
                    found_improvement = True
                    if best_score is None or current_score < best_score:
                        best_score = current_score
                        best_points = current[:]
                    break
            
            temp *= cooling
        
        return current, current_score
    
    # Starting configurations
    starts = []
    
    # 1. 3x4 grid
    starts.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    starts.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 grid
    starts.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. Random points
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if p not in pts:
                pts.append(p)
        starts.append(pts)
    
    # 5. Points designed for many collinearities: e.g., 4 lines of 3
    # Line 1: (0,0), (1,0), (2,0)
    # Line 2: (0,1), (1,1), (2,1)
    # Line 3: (0,2), (1,2), (2,2)
    # Plus 3 more points to make 12
    pts5 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
    for _ in range(3):
        pts5.append([random.randint(-5, 5), random.randint(-5, 5)])
    starts.append([list(p) for p in pts5])
    
    # 6. Pappus-like configuration or other known small point-line configs
    # Try a configuration with 3 lines of 4 points each, sharing points
    # For now, use optimization from various starts
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        try:
            _, _ = optimize_from(start, max_iters=30000)
        except:
            pass
    
    if best_points is None:
        # Fallback to original
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    # Final validation
    if len(best_points) != 12:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    # Check distinctness
    seen = set()
    valid_pts = []
    for p in best_points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            valid_pts.append(p)
    if len(valid_pts) != 12:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    # Check not all collinear
    def all_collinear(pts):
        if len(pts) < 3:
            return True
        (x1, y1), (x2, y2), (x3, y3) = pts[0], pts[1], pts[2]
        area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if area != 0:
            return False
        for p in pts[3:]:
            x4, y4 = p
            area2 = (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)
            if area2 != 0:
                return False
        return True
    
    if all_collinear(best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
