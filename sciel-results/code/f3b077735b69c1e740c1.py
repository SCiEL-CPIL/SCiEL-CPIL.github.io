import time
import random
import itertools
from math import gcd
from collections import defaultdict

DEADLINE = 145  # Safety margin before 150s kill

def _line_key(p1, p2):
    """Return canonical key for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal form: a*x + b*y = c
    # a = -dy, b = dx, c = dx*y1 - dy*x1
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    # Normalize sign: make (a,b) have a>0 or a==0,b>0
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    """
    Builds a map from line_key to list of point indices on that line.
    Also returns a set of all collinear triples (i, j, k) with i<j<k.
    """
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].add(i)
            line_map[key].add(j)
    
    triples = set()
    for key, pts in line_map.items():
        if len(pts) >= 3:
            pts_list = sorted(pts)
            for t in itertools.combinations(pts_list, 3):
                triples.add(t)
    
    return line_map, triples

def _check_no_four_collinear(points):
    """Check that no four points are collinear."""
    _, line_map = _build_lines(points)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return False
    return True

def _max_cap_free(points, triples_set, time_limit=5.0):
    """
    Find max subset with no three collinear using branch and bound.
    Returns the size of the largest such subset.
    """
    n = len(points)
    start_time = time.time()
    
    # triple_conflict[i] = list of (j, k) such that {i, j, k} is a triple
    # We need to check if adding i creates a triple with any existing pair in the subset.
    # Actually, it's easier to check: for each triple {a,b,c}, not all in S.
    # Branch and bound on adding points 0..n-1.
    
    # Precompute for each point i, the set of pairs (j,k) that form a triple with i.
    # If i is added, we cannot have both j and k in S.
    # This is a constraint satisfaction problem.
    
    # Let's use a bitmask for the current subset S.
    # State: (mask, next_idx)
    # Pruning: if |S| + (n - next_idx) <= best, return.
    
    best_size = 0
    best_mask = 0
    
    # For fast validity check:
    # When adding idx to mask, check all triples containing idx.
    # If the other two are already in mask, invalid.
    
    triples_by_point = [set() for _ in range(n)]
    for (i, j, k) in triples_set:
        triples_by_point[i].add((j, k))
        triples_by_point[j].add((i, k))
        triples_by_point[k].add((i, j))
    
    def is_valid_add(mask, idx):
        # Check if adding idx creates a triple
        for (j, k) in triples_by_point[idx]:
            # If j and k are both in mask, then adding idx completes the triple
            if (mask >> j) & 1 and (mask >> k) & 1:
                return False
        return True

    def dfs(mask, next_idx):
        nonlocal best_size, best_mask
        if time.time() - start_time > time_limit:
            return
        
        cur_size = mask.bit_count()
        if cur_size > best_size:
            best_size = cur_size
            best_mask = mask
            
        # Upper bound
        if cur_size + (n - next_idx) <= best_size:
            return
            
        for idx in range(next_idx, n):
            if time.time() - start_time > time_limit:
                return
            # Try including idx
            if is_valid_add(mask, idx):
                dfs(mask | (1 << idx), idx + 1)
            # Try excluding idx
            dfs(mask, idx + 1)

    dfs(0, 0)
    return best_size

def _score(points):
    """Compute the max cap-free subset size."""
    if not _check_no_four_collinear(points):
        return 999  # Invalid
    _, triples = _build_lines(points)
    return _max_cap_free(points, triples, time_limit=8.0)

def _generate_initial_points(seed):
    """
    Generate a set of 20 points.
    We try a few different structured patterns.
    """
    rng = random.Random(seed)
    patterns = []
    
    # Pattern 1: Perturbed 5x4 grid
    pts = []
    for i in range(5):
        for j in range(4):
            x = i * 10 + rng.randint(-3, 3)
            y = j * 10 + rng.randint(-3, 3)
            pts.append([x, y])
    if len(set(tuple(p) for p in pts)) == 20:
        patterns.append(pts)
        
    # Pattern 2: Points on a parabola y = x^2 mod p (scaled)
    # Use p=19, take 20 points? No, 19 points max for distinct x in field.
    # Let's just use random points in a larger box to avoid grid artifacts.
    pts = []
    for _ in range(20):
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        pts.append([x, y])
    if len(set(tuple(p) for p in pts)) == 20:
        patterns.append(pts)
        
    # Pattern 3: Specific known good configuration (if we had one hardcoded)
    # For now, rely on search.
    
    return patterns

def _local_search(points, time_limit=30.0):
    """
    Local search to minimize the score (max cap-free subset size).
    Moves: 
    1. Move a single point to a new location.
    2. Swap two points.
    Constraint: No 4 collinear.
    """
    start_time = time.time()
    current = [p[:] for p in points]
    current_score = _score(current)
    if current_score >= 999:
        return current, current_score
        
    best_points = [p[:] for p in current]
    best_score = current_score
    
    # Candidate locations for moves
    # We'll sample random locations and also neighbors of existing points
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each point
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            ox, oy = current[i]
            
            # Generate candidate new positions
            # 1. Random positions in [-100, 100]
            candidates = []
            for _ in range(10):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                candidates.append((nx, ny))
            
            # 2. Neighbors of other points (to create new collinearities)
            for j in range(20):
                if i == j: continue
                px, py = current[j]
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx = px + dx
                        ny = py + dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            candidates.append((nx, ny))
            
            # Deduplicate candidates
            candidates = list(set(candidates))
            
            for nx, ny in candidates:
                if nx == ox and ny == oy:
                    continue
                current[i] = [nx, ny]
                
                # Check distinctness
                if len(set(tuple(p) for p in current)) != 20:
                    continue
                    
                # Check no 4 collinear (fast check)
                if not _check_no_four_collinear(current):
                    continue
                    
                s = _score(current)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
                    improved = True
                    # Restart search from this better point? 
                    # For simplicity, just continue.
                current[i] = [ox, oy]
                
        if not improved:
            # If no improvement in a full pass, try a random restart or break
            # To avoid getting stuck, we might randomize the current state slightly
            if random.random() < 0.1:
                # Random restart
                i = random.randint(0, 19)
                current[i] = [random.randint(-100, 100), random.randint(-100, 100)]
                if len(set(tuple(p) for p in current)) == 20 and _check_no_four_collinear(current):
                    s = _score(current)
                    if s < best_score:
                        best_score = s
                        best_points = [p[:] for p in current]

    return best_points, best_score

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 999
    
    # Try multiple initial configurations
    for seed in range(50):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        patterns = _generate_initial_points(seed)
        for pts in patterns:
            if not _check_no_four_collinear(pts):
                continue
            
            # Quick score check
            s = _score(pts)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                
            # If promising, run local search
            if s <= 18 and time.time() - start_time < DEADLINE - 30:
                # Allocate time for local search
                remaining = DEADLINE - 25 - (time.time() - start_time)
                if remaining > 5:
                    refined, refined_score = _local_search(pts, time_limit=min(20.0, remaining))
                    if refined_score < best_score:
                        best_score = refined_score
                        best_points = refined

    # Fallback: Parabola (always valid, score 20)
    if best_points is None or best_score >= 20:
        # A simple set of points with no 4 collinear
        # y = x^2 for x in -10..9 has no 4 collinear? 
        # Actually, parabola has no 3 collinear, so score is 20.
        # We need a configuration with lower score.
        # Let's use a known bad configuration (high collinearity) as fallback?
        # No, we need to return a valid one.
        # If we didn't find anything better, return the best we found, even if 20.
        if best_points is None:
            best_points = [[x, x * x] for x in range(-10, 10)]
            
    return best_points
