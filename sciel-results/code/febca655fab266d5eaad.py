"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line (as a normalized pair of points defining it) to count
    # A line is identified by (dx, dy, c) where the line is y = (dy/dx)x + c if dx != 0
    # Better: use a canonical form. For two points (x1,y1),(x2,y2), the line can be
    # represented by (A, B, C) where Ax + By + C = 0, normalized.
    # For integer points, we can use: A = y1-y2, B = x2-x1, C = x1*y2 - x2*y1
    # Then normalize by gcd and sign.
    
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            # Normalize: divide by gcd of |A|,|B|,|C|
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
            if g > 0:
                A //= g
                B //= g
                C //= g
            # Ensure canonical sign: first non-zero of (A,B,C) should be positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # But this counts pairs per line, not points per line.
    # A line with k points has C(k,2) pairs. So if line_counts[key] == 1, that line has exactly 2 points.
    # If line_counts[key] == 3, it could be 3 points (C(3,2)=3) or... no, C(k,2)=3 means k=3.
    # Actually we need to count points on each line, not pairs.
    
    # Let me redo: for each unique line, count how many of the n points lie on it.
    lines = {}  # key -> set of point indices
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, idx_set in lines.items():
        # Verify: count actual points on this line
        A, B, C = key
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        if count == 2:
            ordinary += 1
    return ordinary


def _is_valid(points):
    """Check if points are 12 distinct integer points with |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    # Check not all collinear
    if len(points) >= 3:
        x1, y1 = points[0]
        x2, y2 = points[1]
        for i in range(2, len(points)):
            x3, y3 = points[i]
            if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
                return True
        return False
    return True


def _local_search(points, time_limit):
    """Hill climbing / simulated annealing to reduce ordinary lines."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_cost = _count_ordinary_lines(current)
    best = [list(p) for p in current]
    best_cost = current_cost
    
    # Try moving individual points to nearby positions
    for step in range(100000):
        if time.time() - start_time > time_limit:
            break
        if not _is_valid(current):
            break
        
        i = random.randint(0, 11)
        # Try a few random perturbations
        for _ in range(5):
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_point = [current[i][0] + dx, current[i][1] + dy]
            if abs(new_point[0]) > 10**9 or abs(new_point[1]) > 10**9:
                continue
            # Check distinctness
            new_points = [list(p) for p in current]
            new_points[i] = new_point
            if len(set(tuple(p) for p in new_points)) != 12:
                continue
            new_cost = _count_ordinary_lines(new_points)
            if new_cost < current_cost:
                current = new_points
                current_cost = new_cost
                if current_cost < best_cost:
                    best = [list(p) for p in current]
                    best_cost = current_cost
                break
            elif new_cost == current_cost and random.random() < 0.1:
                # Occasional sideways move
                current = new_points
                current_cost = new_cost
                break
    
    return best, best_cost


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_points = None
    best_cost = float('inf')
    
    # Strategy 1: Try structured grids and variations
    # 3x4 grid
    candidates = []
    
    # Basic 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    # 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # Diagonal-shifted grids
    for shift in range(-2, 3):
        pts = []
        for y in range(3):
            for x in range(4):
                pts.append([x + shift * y, y])
        candidates.append(pts)
    
    # Near-grid with one point moved
    base = [[x, y] for x in range(4) for y in range(3)]
    for i in range(12):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                pts = [list(p) for p in base]
                pts[i] = [pts[i][0] + dx, pts[i][1] + dy]
                if len(set(tuple(p) for p in pts)) == 12:
                    candidates.append(pts)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(cand):
            continue
        cost = _count_ordinary_lines(cand)
        if cost < best_cost:
            best_cost = cost
            best_points = [list(p) for p in cand]
    
    # Strategy 2: Local search from best candidate
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 5:
            ls_points, ls_cost = _local_search(best_points, remaining - 5)
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_points = ls_points
    
    # Strategy 3: Randomized restarts with local search
    while time.time() < deadline - 5:
        # Generate random-ish structured points
        # Use small integer coordinates to keep things manageable
        scale = random.choice([1, 2, 3])
        pts = []
        for _ in range(12):
            x = random.randint(-scale * 3, scale * 3)
            y = random.randint(-scale * 3, scale * 3)
            pts.append([x, y])
        if len(set(tuple(p) for p in pts)) != 12:
            continue
        if not _is_valid(pts):
            continue
        cost = _count_ordinary_lines(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = [list(p) for p in pts]
        
        # Quick local search
        remaining = deadline - time.time()
        if remaining > 3:
            ls_points, ls_cost = _local_search(pts, min(remaining - 3, 5))
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_points = ls_points
    
    # Strategy 4: Try known good configurations
    # The 12-point configuration from the orchard-planting problem
    # Try a configuration based on a 3x4 grid with strategic perturbations
    
    if best_points is None:
        # Fallback: return the 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_cost = _count_ordinary_lines(best_points)
    
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
