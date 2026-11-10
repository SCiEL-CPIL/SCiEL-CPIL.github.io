"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations
from fractions import Fraction

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def check_no_three_collinear(points):
    """Check no three points are collinear. points is list of (x, y) tuples with integer coords."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def check_no_four_concyclic(points):
    """Check no four points are concyclic. points is list of (x, y) tuples with integer coords."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    b2 = a2 - a1
                    c2 = x2 - x1
                    d2 = y2 - y1
                    b3 = a3 - a1
                    c3 = x3 - x1
                    d3 = y3 - y1
                    b4 = a4 - a1
                    c4 = x4 - x1
                    d4 = y4 - y1
                    det3 = b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)
                    if det3 == 0:
                        return False
    return True

def check_dist_sq_is_square(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dsq = dx*dx + dy*dy
    return is_perfect_square(dsq)

def is_valid_set(points):
    """Check all constraints for a set of integer points."""
    n = len(points)
    if n < 3:
        return False
    pts_set = set(points)
    if len(pts_set) != n:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if not check_dist_sq_is_square(points[i], points[j]):
                return False
    if not check_no_three_collinear(points):
        return False
    if not check_no_four_concyclic(points):
        return False
    return True

def generate_candidates(max_coord=800):
    """Generate candidate points with integer distances to origin using Pythagorean triples."""
    candidates = set()
    for m in range(2, int(math.sqrt(max_coord*max_coord)) + 2):
        for n in range(1, m):
            if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                if c <= max_coord:
                    for sa in [-1, 1]:
                        for sb in [-1, 1]:
                            candidates.add((sa*a, sb*b))
                            candidates.add((sa*b, sb*a))
    # Points on axes
    for r in range(1, max_coord+1):
        candidates.add((r, 0))
        candidates.add((-r, 0))
        candidates.add((0, r))
        candidates.add((0, -r))
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Known 5-point solution
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = len(best_points)
    
    # Generate candidates
    candidates = generate_candidates(800)
    # Sort by distance to origin for deterministic ordering
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    # Strategy 1: Greedy extension from known 5-point set
    current = list(best_points)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current:
            continue
        # Quick check: integer distance to all current points
        valid = True
        for px, py in current:
            dx = cand[0] - px
            dy = cand[1] - py
            if not is_perfect_square(dx*dx + dy*dy):
                valid = False
                break
        if not valid:
            continue
        # Full validity check
        new_set = current + [cand]
        if is_valid_set(new_set):
            current.append(cand)
            if len(current) > best_count:
                best_count = len(current)
                best_points = list(current)
    
    # Strategy 2: Randomized search with different starting points
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current = list(best_points)
        elif restart == 1:
            current = [(0, 0), (0, 5), (4, 0)]
        elif restart == 2:
            current = [(0, 0), (3, 4), (5, 0)]
        elif restart == 3:
            current = [(0, 0), (5, 12), (0, 13)]
        elif restart == 4:
            current = [(0, 0), (8, 15), (17, 0)]
        else:
            # Random start with 3-4 points
            current = [(0, 0)]
            for _ in range(random.randint(2, 3)):
                if time.time() > deadline:
                    break
                attempts = 0
                while attempts < 3000:
                    attempts += 1
                    m = random.randint(2, 80)
                    n = random.randint(1, m-1)
                    if (m - n) % 2 == 0 or math.gcd(m, n) != 1:
                        continue
                    a = m*m - n*n
                    b = 2*m*n
                    if random.random() < 0.5:
                        cand = (a, b)
                    else:
                        cand = (b, a)
                    if random.random() < 0.5:
                        cand = (-cand[0], cand[1])
                    if random.random() < 0.5:
                        cand = (cand[0], -cand[1])
                    if cand in current:
                        continue
                    valid = True
                    for px, py in current:
                        dx = cand[0] - px
                        dy = cand[1] - py
                        if not is_perfect_square(dx*dx + dy*dy):
                            valid = False
                            break
                    if valid and check_no_three_collinear(current + [cand]):
                        current.append(cand)
                        break
        
        # Try to extend current set
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in current:
                continue
            valid = True
            for px, py in current:
                dx = cand[0] - px
                dy = cand[1] - py
                if not is_perfect_square(dx*dx + dy*dy):
                    valid = False
                    break
            if not valid:
                continue
            new_set = current + [cand]
            if is_valid_set(new_set):
                current = new_set
                if len(current) > best_count:
                    best_count = len(current)
                    best_points = list(current)
    
    # Strategy 3: Local search - try to swap points
    if time.time() < deadline - 20:
        for iteration in range(80):
            if time.time() > deadline:
                break
            current = list(best_points)
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for cand in candidates:
                    if cand in current:
                        continue
                    new_set = current[:i] + current[i+1:] + [cand]
                    if is_valid_set(new_set):
                        if len(new_set) > len(best_points):
                            best_points = new_set
                            best_count = len(best_points)
                        improved = True
                        break
            if not improved:
                break
    
    # Final verification
    if is_valid_set(best_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback to known good set
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[-56, 1], [33, 1]], [[-16, 1], [63, 1]], [[0, 1], [33, 1]], [[16, 1], [63, 1]]]}
# EVOLVE-BLOCK-END
