import time
import random
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _normalize_line(p1, p2):
    """Return a canonical representation of the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    # Line: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    # Normalize
    if a == 0 and b == 0:
        return (0, 0, 0)
    # Make the first non-zero coefficient positive
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    # Reduce by gcd
    from math import gcd
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    return (a, b, c)

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _normalize_line(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    """Check that not all points are collinear."""
    if len(points) < 3:
        return True
    p1, p2, p3 = points[0], points[1], points[2]
    return (p2[0]-p1[0])*(p3[1]-p1[1]) != (p3[0]-p1[0])*(p2[1]-p1[1])

def _local_search(points, max_iterations=50000, time_limit=30):
    """Local search: randomly move one point to a nearby integer location."""
    best_score = _count_ordinary_lines(points)
    best_points = [list(p) for p in points]
    rng = random.Random(42)
    start_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        # Pick a random point to move
        idx = rng.randrange(len(points))
        old_x, old_y = best_points[idx]
        # Move to a random nearby location
        new_x = old_x + rng.randint(-5, 5)
        new_y = old_y + rng.randint(-5, 5)
        if new_x == old_x and new_y == old_y:
            continue
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        # Check if new location is already occupied
        new_pt = [new_x, new_y]
        if any(p == new_pt for i, p in enumerate(best_points) if i != idx):
            continue
        best_points[idx] = new_pt
        if not _not_all_collinear(best_points):
            best_points[idx] = [old_x, old_y]
            continue
        score = _count_ordinary_lines(best_points)
        if score < best_score:
            best_score = score
            best_points[idx] = new_pt
            if best_score <= 6:
                break
        else:
            best_points[idx] = [old_x, old_y]
    
    return [list(p) for p in best_points], best_score

def _generate_structured_candidates():
    """Generate structured candidates based on known configurations."""
    candidates = []
    
    # 3x4 grid (current best)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # Points on a conic: parabola y = x^2 for x in some range
    candidates.append([[x, x*x] for x in range(-3, 4)])  # 7 points, need 12
    
    # Mix of structured patterns
    # 3 points on each of 4 parallel lines
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        pts = []
        for i in range(4):
            base_x = i * 10
            base_y = i * 5 if dy != 0 else 0
            for j in range(3):
                pts.append([base_x + j * dx * 3, base_y + j * dy * 3])
        if len(pts) == 12 and _not_all_collinear(pts):
            candidates.append(pts)
    
    # Elliptic curve inspired: points with y^2 = x^3 + ax + b type structure
    # Use small integer points that might have high collinearity
    pts = []
    for x in range(-2, 4):
        for y in range(-2, 3):
            pts.append([x, y])
            if len(pts) == 12:
                break
        if len(pts) == 12:
            break
    if len(pts) == 12 and _not_all_collinear(pts):
        candidates.append(pts)
    
    # Points from known Sylvester-Gallai extremal examples
    # Boros-Furedi type: 3 groups of 4 on 3 concurrent lines
    for angle_offset in [0, 1, 2]:
        pts = []
        for group in range(3):
            base_angle = group * 2 + angle_offset
            for k in range(4):
                x = (k + 1) * 3
                y = k * (group + 1)
                pts.append([x, y])
        if len(pts) == 12 and _not_all_collinear(pts):
            candidates.append(pts)
    
    return candidates

def construct():
    deadline = time.time() + 130  # 20s safety margin
    rng = random.Random(12345)
    
    best_points = None
    best_score = float('inf')
    
    # Step 1: Evaluate structured candidates
    candidates = _generate_structured_candidates()
    for cand in candidates:
        if time.time() > deadline - 60:
            break
        if len(cand) != 12:
            continue
        # Check distinctness
        if len(set(tuple(p) for p in cand)) != 12:
            continue
        if not _not_all_collinear(cand):
            continue
        score = _count_ordinary_lines(cand)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in cand]
            if best_score <= 6:
                return best_points
    
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    # Step 2: Local search from best structured candidate
    if time.time() < deadline - 20:
        improved_points, improved_score = _local_search(
            best_points, 
            max_iterations=80000, 
            time_limit=int(deadline - time.time() - 25)
        )
        if improved_score < best_score:
            best_points = improved_points
            best_score = improved_score
    
    # Step 3: Random restarts with local search
    if best_score > 6 and time.time() < deadline - 30:
        for attempt in range(30):
            if time.time() > deadline - 25:
                break
            # Generate random points in a reasonable range
            pts = []
            seen = set()
            while len(pts) < 12:
                x = rng.randint(-20, 20)
                y = rng.randint(-20, 20)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            if not _not_all_collinear(pts):
                continue
            score = _count_ordinary_lines(pts)
            if score < best_score:
                best_score = score
                best_points = [list(p) for p in pts]
                if best_score <= 6:
                    break
            # Local search from this random start
            if time.time() < deadline - 30:
                remaining = int(deadline - time.time() - 25)
                if remaining > 5:
                    lp, ls = _local_search(pts, max_iterations=30000, time_limit=remaining)
                    if ls < best_score:
                        best_score = ls
                        best_points = lp
                        if best_score <= 6:
                            break
    
    return best_points
# EVOLVE-BLOCK-END
