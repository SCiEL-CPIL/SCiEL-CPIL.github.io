"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    # Map each line to a canonical tuple; count points per line
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            c = -(dx * y1 + dy * x1)
            g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
            if g2 > 0:
                dx //= g2
                dy //= g2
                c //= g2
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy, c = -dx, -dy, -c
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    # line_counts[key] = number of pairs on that line = C(k,2) where k = points on line
    # We need to recover k from C(k,2) = m: k = (1 + sqrt(1+8m))/2
    ordinary = 0
    for key, m in line_counts.items():
        # m = C(k,2) => k^2 - k - 2m = 0 => k = (1 + sqrt(1+8m)) / 2
        disc = 1 + 8 * m
        import math
        sqrt_d = math.isqrt(disc)
        if sqrt_d * sqrt_d == disc:
            k = (1 + sqrt_d) // 2
            if k == 2:
                ordinary += 1
            # else k >= 3, not ordinary
        else:
            # Should not happen for valid inputs
            ordinary += 1
    return ordinary

def _eval_config(points):
    """Return (num_ordinary, valid) for a configuration of 12 points."""
    if len(points) != 12:
        return (999, False)
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return (999, False)
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (999, False)
    # Check not all collinear
    if len(set(_line_key(points[0], points[1]) for _ in [0])) == 1:
        # All on same line?
        pass
    # More careful: check if all 12 are collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    all_col = True
    for x, y in points[2:]:
        if dx * (y - y0) != dy * (x - x0):
            all_col = False
            break
    if all_col:
        return (999, False)
    return (_count_ordinary(points), True)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    c = -(dx * y1 + dy * x1)
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy, c = -dx, -dy, -c
    return (dx, dy, c)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_points)
    
    # Try several structured configurations
    candidates = []
    
    # 1. 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 5x5 grid subset: pick 12 points strategically
    # Try center-heavy selection
    for offset in [(-1, -1), (0, 0), (1, 1)]:
        ox, oy = offset
        # Pick from 5x5 grid centered
        pts = []
        for x in range(5):
            for y in range(5):
                pts.append([x + ox, y + oy])
        # Try different subsets of 12 from this 25-point set
        # For now, just try the full 5x5 minus 13 points
        # Actually, let's try specific patterns
    
    # 4. Try a "cross" pattern: points on x-axis, y-axis, and diagonals
    # 5 points on x-axis: (-2,-1,0,1,2), 5 on y-axis, but (0,0) shared
    # That's 9 points. Add 3 more.
    
    # 5. Try configuration based on 3 lines with 4 points each, plus 3 more
    # Line y=0: (0,0),(1,0),(2,0),(3,0)
    # Line y=1: (0,1),(1,1),(2,1),(3,1)  
    # Line y=2: (0,2),(1,2),(2,2),(3,2)
    # That's the 3x4 grid again.
    
    # 6. Try a hexagonal/symmetric arrangement
    # 7 points: center + 6 around it (hexagon)
    # Plus 5 more
    
    # Let me try a more systematic local search on the 3x4 grid
    # Start with 3x4 grid and try moving points to reduce ordinary lines
    
    # First, evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score, valid = _eval_config(cand)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    # Local search: try swapping/moving individual points
    # Represent points as list of [x,y]
    current = [p[:] for p in best_points]
    
    # Search space: for each point, try moving it to nearby positions
    # and also try swapping with other points
    search_radius = 5  # max coordinate change
    
    improved = True
    max_iter = 5000
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(12):
            if time.time() > deadline:
                break
            x, y = current[i]
            # Try small perturbations
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check if this point already exists
                    trial = [p[:] for p in current]
                    trial[i] = [new_x, new_y]
                    # Check distinctness
                    if len(set(tuple(p) for p in trial)) != 12:
                        continue
                    score, valid = _eval_config(trial)
                    if valid and score < best_score:
                        best_score = score
                        current = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Also try a second phase: larger moves
    if time.time() < deadline - 5:
        current = [p[:] for p in best_points]
        improved = True
        for iteration in range(3000):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current[i]
                # Try random-ish larger moves
                for dx in list(range(-15, 16, 3)):
                    for dy in list(range(-15, 16, 3)):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        trial = [p[:] for p in current]
                        trial[i] = [new_x, new_y]
                        if len(set(tuple(p) for p in trial)) != 12:
                            continue
                        score, valid = _eval_config(trial)
                        if valid and score < best_score:
                            best_score = score
                            current = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
