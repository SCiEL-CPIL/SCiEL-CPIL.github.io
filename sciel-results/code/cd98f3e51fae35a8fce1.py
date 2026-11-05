"""
Optimized search for 16 integer points maximizing lines with exactly 4 points.
Uses Simulated Annealing with pairwise swaps and NumPy vectorized evaluation.
"""
import time
import random
import numpy as np
from math import exp

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _evaluate_fast(points):
    """
    Evaluate a list of 16 [x,y] points.
    Returns (is_valid, score).
    is_valid: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear.
    score: number of lines containing exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    
    # Convert to numpy arrays for fast operations
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Check bounds
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
        
    # Check distinctness
    if len(set(map(tuple, pts.tolist()))) != 16:
        return False, 0

    # Count pairs per line
    # Use vectorized line keys
    iu, ju = np.triu_indices(16, k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    
    # gcd
    ax = np.abs(dx)
    ay = np.abs(dy)
    g = np.gcd(ax, ay)
    
    # avoid division by zero: where g==0 (both 0) can't happen since distinct points
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    
    # canonicalize sign: make first nonzero positive
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    
    # c = dyn*x1 - dxn*y1
    c = dyn * xs[iu] - dxn * ys[iu]
    
    keys = np.column_stack((dxn, dyn, c))
    
    # Count occurrences
    # Using a dictionary or sorting. For 120 items, sorting is fast.
    # Or use a hash map via python dict on tuples
    keys_t = list(map(tuple, keys.tolist()))
    
    # Use a simple frequency counter
    counts = {}
    for k in keys_t:
        counts[k] = counts.get(k, 0) + 1
        
    score = 0
    max_count = 0
    for count in counts.values():
        if count > max_count:
            max_count = count
        if count == 6:  # exactly 4 points -> 6 pairs
            score += 1
            
    if max_count >= 10:  # 5 or more points on a line
        return False, 0
        
    return True, score

def _random_initial():
    """Generate a random valid set of 16 points."""
    pts = set()
    while len(pts) < 16:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        pts.add((x, y))
    return [[x, y] for x, y in pts]

def construct():
    best_score = -1
    best_points = None
    deadline = _START_TIME + _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with the 4x4 grid as a baseline
    grid = [[x, y] for x in range(4) for y in range(4)]
    is_valid, score = _evaluate_fast(grid)
    if is_valid and score > best_score:
        best_score = score
        best_points = [list(p) for p in grid]
        
    current_points = [list(p) for p in best_points] if best_points else _random_initial()
    current_valid, current_score = _evaluate_fast(current_points)
    if not current_valid:
        current_points = _random_initial()
        current_valid, current_score = _evaluate_fast(current_points)
        
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Simulated Annealing loop
    while time.time() < deadline:
        # Pick two distinct indices to swap
        i, j = random.sample(range(16), 2)
        
        old_i = current_points[i]
        old_j = current_points[j]
        
        # Perform swap
        current_points[i] = old_j
        current_points[j] = old_i
        
        new_valid, new_score = _evaluate_fast(current_points)
        
        if new_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
                
            # Accept if better, or with probability exp((new-old)/T)
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
            else:
                # Accept worse move with probability
                p = exp(delta / T)
                if random.random() < p:
                    current_score = new_score
                else:
                    # Revert
                    current_points[i] = old_i
                    current_points[j] = old_j
        else:
            # Revert if invalid
            current_points[i] = old_i
            current_points[j] = old_j
            
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally restart to escape local minima
        if random.random() < 0.01:
            new_start = _random_initial()
            is_valid, score = _evaluate_fast(new_start)
            if is_valid:
                current_points = new_start
                current_score = score
                T = 10.0 # Reset temperature
                
    return best_points
