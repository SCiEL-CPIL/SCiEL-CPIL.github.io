"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    offset = dy * x1 - dx * y1
    return (dx, dy, offset)

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _eval_config(points):
    """Evaluate a configuration: return (ordinary_count, is_valid)."""
    n = len(points)
    if n != 12:
        return (float('inf'), False)
    
    if len(set(map(tuple, points))) != 12:
        return (float('inf'), False)
    
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (float('inf'), False)
    
    if n >= 3:
        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            all_collinear = True
            for i in range(3, n):
                xi, yi = points[i]
                c = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
                if c != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return (float('inf'), False)
    
    ordinary = _count_ordinary(points)
    return (ordinary, True)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Strategy: Exhaustive/Heuristic search over a small dense grid.
    # We assume that optimal or near-optimal configurations for small n
    # can be found within a small integer grid.
    
    # Define a candidate grid. 
    # A 7x7 grid centered at 0 has 49 points.
    # Choosing 12 points is hard to exhaust, so we use local search.
    
    grid_size = 4  # Range -4 to 4 -> 9x9 = 81 points
    # Actually, let's use a slightly larger range to allow for non-grid optimal shapes,
    # but keep it small for search efficiency.
    # Let's try -5 to 5 -> 11x11 = 121 points.
    
    lo = -5
    hi = 5
    grid_points = [[x, y] for x in range(lo, hi + 1) for y in range(lo, hi + 1)]
    
    # Pre-compute line incidences for the grid to speed up evaluation?
    # For 121 points, pairs are ~7000. 
    # We can pre-compute a map: point_index -> list of other point indices collinear with it.
    # But we need to count lines with EXACTLY 2 points from the SUBSET.
    
    # Let's use a fast local search.
    
    best_points = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. 3x4 grid
    cand1 = [[x, y] for x in range(4) for y in range(3)]
    # 2. 4x3 grid
    cand2 = [[x, y] for x in range(3) for y in range(4)]
    # 3. 2x6 grid
    cand3 = [[x, y] for x in range(6) for y in range(2)]
    # 4. Random subset from grid
    random.seed(42)
    for _ in range(20):
        cand = random.sample(grid_points, 12)
        # Check validity
        s, v = _eval_config(cand)
        if v and s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
            
    for cand in [cand1, cand2, cand3]:
        s, v = _eval_config(cand)
        if v and s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
            
    if best_points is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(3)]
        
    # Local Search: Swap points
    # To speed up, we can try to improve by moving one point to a better location.
    # We will try all possible moves for each point if time permits, 
    # or random moves with many iterations.
    
    # Since we are in a small grid, we can try a "greedy" improvement:
    # For each point in the current best, try replacing it with every other point in the grid.
    # This is 12 * 121 = 1452 evaluations per pass.
    # Each evaluation is O(12^2) = 144 line key calculations.
    # 1452 * 144 ~ 200,000 ops. Very fast.
    # We can do many passes.
    
    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        current = [p[:] for p in best_points]
        
        for i in range(12):
            if time.time() > deadline:
                break
            old_point = current[i]
            # Try replacing point i with every other point in the grid
            for new_point in grid_points:
                if new_point == old_point:
                    continue
                # Check if new_point is already in current (excluding i)
                # We can use a set for fast lookup
                current_set = set(map(tuple, current))
                current_set.remove(tuple(old_point))
                if tuple(new_point) in current_set:
                    continue
                
                trial = current[:]
                trial[i] = new_point
                score, valid = _eval_config(trial)
                if valid and score < best_score:
                    best_score = score
                    best_points = trial
                    current = trial
                    current_set = set(map(tuple, current))
                    improved = True
                    # Don't break, keep trying to see if we can improve further in this pass?
                    # Or restart pass? Restarting is safer for convergence.
                    # Let's continue with the new best for subsequent i's.
            
            if time.time() > deadline:
                break
        
        # If no improvement in a full pass, try random restarts from current best
        if not improved:
             # Try a few random perturbations
             for _ in range(100):
                 if time.time() > deadline: break
                 trial = [p[:] for p in best_points]
                 idx = random.randint(0, 11)
                 new_p = random.choice(grid_points)
                 # Ensure distinct
                 if new_p == trial[idx]: continue
                 if new_p in trial: continue
                 trial[idx] = new_p
                 s, v = _eval_config(trial)
                 if v and s < best_score:
                     best_score = s
                     best_points = trial
                     improved = True
                     break
                     
    return best_points
# EVOLVE-BLOCK-END
