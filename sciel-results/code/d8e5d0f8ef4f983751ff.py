import time
import random
import math
from collections import defaultdict
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return (0, 0, 1) if C != 0 else (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_4_lines(points):
    """
    Returns (count_4, is_valid, line_map)
    line_map: dict mapping line_key -> set of point_indices on that line
    """
    n = len(points)
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_points(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
            
    count_4 = 0
    is_valid = True
    for key, indices in line_map.items():
        cnt = len(indices)
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            is_valid = False
    return count_4, is_valid, line_map

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Generate a large pool of candidate points
    # We want points that are likely to form 4-lines.
    # A 4-line requires 4 collinear points.
    # Let's generate points on many lines.
    # Strategy: Generate a grid of points, and also some points on random lines.
    # But we need to ensure we can pick 16 distinct points.
    
    # Let's start with a 10x10 grid of points (100 points)
    # And maybe some other points.
    # Actually, the 4x4 grid is a subset of a larger grid.
    # If we search over a 10x10 grid, we can find better configurations.
    
    # Pool: All points (x, y) with -5 <= x <= 5, -5 <= y <= 5
    # This is 11x11 = 121 points.
    # We will select 16 from this pool.
    
    pool = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            pool.append([x, y])
    
    # Add some points outside this grid to allow for more variety?
    # Maybe a 15x15 grid? 225 points.
    # Let's stick to 11x11 for now to keep selection search manageable.
    # If we have 121 candidates, selecting 16 is C(121, 16) which is huge.
    # We will use Simulated Annealing to select the 16 points.
    
    # Initial selection: Random 16 points from pool?
    # Or start with the 4x4 grid points (which are in the pool if we shift/scale).
    # The 4x4 grid is [0..3]x[0..3]. This is in the pool.
    # Let's start with the 4x4 grid points.
    
    initial_indices = []
    for x in range(0, 4):
        for y in range(0, 4):
            # Find index in pool
            # Pool index: (x+5)*11 + (y+5)
            idx = (x + 5) * 11 + (y + 5)
            initial_indices.append(idx)
            
    current_selection = initial_indices[:]
    current_points = [pool[i] for i in current_selection]
    current_score, current_valid, _ = _count_4_lines(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    if not current_valid:
        # Should not happen for 4x4 grid
        return best_points

    random.seed(42)
    
    # Simulated Annealing
    # State: selection of 16 indices from pool (size 121)
    # Move: Swap one selected index with one unselected index.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random index to swap out
        out_idx_pos = random.randint(0, 15)
        out_pool_idx = current_selection[out_idx_pos]
        
        # Pick a random index to swap in (not currently selected)
        in_pool_idx = random.randint(0, len(pool) - 1)
        while in_pool_idx in current_selection:
            in_pool_idx = random.randint(0, len(pool) - 1)
            
        # Apply swap
        new_selection = current_selection[:]
        new_selection[out_idx_pos] = in_pool_idx
        
        # Check for duplicates (should be handled by while loop, but just in case)
        if len(set(new_selection)) != 16:
            continue
            
        new_points = [pool[i] for i in new_selection]
        
        new_score, new_valid, _ = _count_4_lines(new_points)
        
        if not new_valid:
            # Invalid configuration (5 collinear). Reject.
            # Or penalize heavily?
            # Let's just reject to keep search space valid.
            continue
            
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_selection = new_selection
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in new_points]
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_selection = new_selection
                current_points = new_points
                current_score = new_score
                
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasionally restart with a random selection to escape local minima?
        # If we are stuck for a long time.
        # Simple restart: if steps % 1000 == 0 and T is low, reheat.
        if steps % 500 == 0:
            T = max(T, 0.5)

    return best_points
