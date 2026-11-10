import random
import time
import sys
from itertools import combinations

_START_TIME = None
DEADLINE = 150.0
MARGIN = 15.0

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _deadline_left():
    return DEADLINE - MARGIN - _elapsed()

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def count_collinear_triples(points):
    """Count the number of collinear triples in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if invalid."""
    n = len(points)
    # For each pair, count points on the line
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def build_collinear_triples(points):
    """Return list of (i, j, k) tuples for collinear triples."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n

    collinear_triples = build_collinear_triples(points)
    
    if not collinear_triples:
        return n

    # pair_to_third[i][j] = set of k such that {i, j, k} is collinear
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j: continue
            pair_to_third[i][j] = set()
            
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    full_mask = (1 << n) - 1

    # Precompute popcount for masks up to 2^20? No, just use bin().count or int.bit_count
    # Python 3.10+ has int.bit_count(). Assuming modern Python.
    # If not, fallback. Let's assume 3.10+ for speed.
    try:
        (1).bit_count()
        def popcount(x): return x.bit_count()
    except AttributeError:
        def popcount(x): return bin(x).count('1')

    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Pruning: if current_size + popcount(remaining) <= best_size, prune
        if current_size + popcount(remaining_mask) <= best_size:
            return

        # Pick a point from remaining
        # Heuristic: Pick the point with the most conflicts? 
        # Or just the lowest index.
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
                
        if p == -1:
            return

        # Option 1: Exclude p
        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))

        # Option 2: Include p
        # Check if p is compatible with current_mask
        compatible = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
            
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
                
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)

    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_triple_count = -1
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    # Strategy: 
    # 1. Generate diverse initial candidates using grid structures to maximize collinear triples.
    # 2. Repair to ensure no 4 collinear.
    # 3. Local search to maximize triple count (proxy for minimizing score).
    # 4. Periodically evaluate the exact score of the best candidates to guide search?
    #    Exact score is expensive (B&B). We might only evaluate the final best few.
    #    However, triple count is a good proxy. More triples -> likely smaller max independent set.
    
    # To improve over the current 1.4286, we need to find configurations with more collinear triples.
    # The current program uses a 6x4 grid. Let's try different grid dimensions or shapes.
    
    grid_shapes = [
        (6, 4), # 24 points
        (5, 5), # 25 points
        (10, 3), # 30 points
        (8, 4), # 32 points
        (7, 4), # 28 points
        (4, 6), # 24 points
    ]
    
    for restart in range(500):
        if _deadline_left() < 5.0:
            break
            
        # Choose a random grid shape
        rows, cols = random.choice(grid_shapes)
        
        ox = random.randint(-80, 80)
        oy = random.randint(-80, 80)
        sx = random.randint(1, 10)
        sy = random.randint(1, 10)
        
        # Add some jitter to grid points to break perfect grid symmetries that might cause 4-collinear
        # or to explore nearby space
        candidates = []
        for i in range(rows):
            for j in range(cols):
                x = ox + i * sx + random.randint(-2, 2)
                y = oy + j * sy + random.randint(-2, 2)
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.append((x, y))
                    
        # Ensure we have at least 20 unique candidates
        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        while len(unique_candidates) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        random.shuffle(unique_candidates)
        points = unique_candidates[:20]
        
        # Ensure no 4 collinear
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            if not has_four_collinear(points):
                valid = True
                break
            # Find a violating point and move it
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
            attempts += 1
            
        if not valid:
            continue
            
        current_triple_count = count_collinear_triples(points)
        
        if current_triple_count > best_triple_count:
            best_triple_count = current_triple_count
            best_points = points.copy()
            
        # Local Search: Maximize collinear triples
        current_points = points.copy()
        current_tc = current_triple_count
        
        # Simulated Annealing style or Hill Climbing
        # Let's do a few steps of hill climbing
        steps = 100 if restart < 100 else 50
        
        for step in range(steps):
            if _deadline_left() < 10.0:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            # Try moving to a nearby point or a random point
            if random.random() < 0.5:
                # Local move
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_p = (old_p[0] + dx, old_p[1] + dy)
                if new_p[0] < -100 or new_p[0] > 100 or new_p[1] < -100 or new_p[1] > 100:
                    continue
            else:
                new_p = random_point()
            
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            if has_four_collinear(candidate_points):
                continue
                
            new_tc = count_collinear_triples(candidate_points)
            
            # Accept if better or with small probability (SA)
            if new_tc > current_tc:
                current_points = candidate_points
                current_tc = new_tc
                if current_tc > best_triple_count:
                    best_triple_count = current_triple_count
                    best_points = current_points.copy()
            else:
                # Simple hill climbing: only accept if equal? No, strict improvement.
                # Let's allow equal to wander?
                if new_tc == current_tc and random.random() < 0.1:
                    current_points = candidate_points

    if best_points is None:
        # Fallback: Return any valid set
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(100):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points

    # Optional: If we have time, evaluate the exact score of the best point and maybe try to improve it specifically?
    # The score is the largest subset with no 3 collinear.
    # Minimizing this is equivalent to maximizing the "density" of collinear triples in a way that blocks large independent sets.
    # Our proxy (total triple count) is good but not perfect.
    # Given time constraints, we rely on the proxy.
    
    return [list(p) for p in best_points]
