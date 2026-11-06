"""
Optimized solver for the 16-point integer collinearity problem.
Uses Simulated Annealing to maximize the number of lines with exactly 4 points.
"""
import time
import random
import math
from math import gcd

_DEADLINE = 110.0  # Safety margin before 120s limit

def _line_key(p1, p2):
    """
    Returns a canonical tuple (A, B, C) representing the line Ax + By = C.
    Normalization:
    1. A, B are integers.
    2. gcd(A, B) = 1.
    3. A > 0, or (A == 0 and B > 0).
    """
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    # If points are identical, this shouldn't happen in a valid set, 
    # but handle it gracefully to avoid ZeroDivisionError or (0,0) key issues.
    if dx == 0 and dy == 0:
        return None

    # Line equation: dy * (x - x1) - dx * (y - y1) = 0
    # dy * x - dx * y = dy * x1 - dx * y1
    A = dy
    B = -dx
    C = A * x1 + B * y1
    
    # Normalize
    g = gcd(abs(A), abs(B))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    # Enforce canonical sign
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate(points):
    """
    Evaluates the configuration.
    Returns (score, valid).
    Score: number of lines with exactly 4 points.
    Valid: True if no 5 collinear, distinct, and within bounds.
    """
    n = len(points)
    if n != 16:
        return -1, False
    
    # Check bounds and distinctness
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False
        pt = (x, y)
        if pt in seen:
            return -1, False
        seen.add(pt)
        
    line_map = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                return -1, False
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    exact_4_count = 0
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) > 4:
            return -1, False
        if len(pts_in_line) == 4:
            exact_4_count += 1
            
    return exact_4_count, True

def _grid_start():
    """4x4 grid centered at origin."""
    pts = []
    for x in range(-1, 3):
        for y in range(-1, 3):
            pts.append([x, y])
    return pts

def _random_start():
    """Random points within bounds."""
    return [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]

def _simulate_annealing(start_points, time_limit, seed=None):
    """
    Runs Simulated Annealing.
    """
    if seed is not None:
        random.seed(seed)
        
    current = [list(p) for p in start_points]
    current_score, valid = _evaluate(current)
    
    if not valid:
        # Try to fix initial state with small jitters if invalid
        for _ in range(100):
            idx = random.randint(0, 15)
            current[idx] = [current[idx][0] + random.randint(-2, 2), 
                            current[idx][1] + random.randint(-2, 2)]
            current_score, valid = _evaluate(current)
            if valid:
                break
        if not valid:
            # Fallback to grid if we can't fix it
            current = _grid_start()
            current_score, valid = _evaluate(current)
            if not valid:
                return current, 0 # Should not happen

    best = [list(p) for p in current]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    
    start_time = time.time()
    
    while time.time() < time_limit:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.7:
            # Jitter: small move
            idx = random.randint(0, 15)
            delta_x = random.randint(-3, 3)
            delta_y = random.randint(-3, 3)
            if delta_x == 0 and delta_y == 0:
                continue
            new_pt = [current[idx][0] + delta_x, current[idx][1] + delta_y]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            trial = [list(p) for p in current]
            trial[idx] = new_pt
        elif move_type < 0.9:
            # Jump: random move
            idx = random.randint(0, 15)
            new_pt = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            trial = [list(p) for p in current]
            trial[idx] = new_pt
        else:
            # Slide: Move a point towards a line defined by two others?
            # Simplified: Pick 3 points, try to align the 4th with 2 of them?
            # This is complex. Let's do a "Reflect" move.
            idx = random.randint(0, 15)
            # Reflect across x=0 or y=0 or another point?
            # Reflect across origin
            new_pt = [-current[idx][0], -current[idx][1]]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            trial = [list(p) for p in current]
            trial[idx] = new_pt

        trial_score, trial_valid = _evaluate(trial)
        
        if trial_valid:
            delta = trial_score - current_score
            if delta > 0:
                # Accept
                current = trial
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best = [list(p) for p in current]
            else:
                # Metropolis criterion
                if random.random() < math.exp(delta / T):
                    current = trial
                    current_score = trial_score
        else:
            # Invalid state: reject unless we are very hot? 
            # Better to just reject to keep search in valid space mostly.
            # Or accept with very low prob to escape? 
            # Let's reject invalid states to maintain structural integrity.
            pass
        
        T = max(T * cooling, T_min)
        
    return best, best_score

def construct():
    start_time = time.time()
    remaining = _DEADLINE - start_time
    
    best_points = None
    best_score = -1
    
    # Define multiple starting configurations
    starts = [
        _grid_start(),
        _random_start(),
        # A "thick" grid: 5x4 minus 4 points?
        [[x, y] for x in range(-2, 3) for y in range(-1, 3)], # 10 points
        # Add 6 random points
    ]
    
    # Generate more diverse starts
    for i in range(5):
        random.seed(42 + i)
        starts.append(_random_start())
        
    # Run SA on each start
    # Allocate time proportionally or evenly
    num_starts = len(starts)
    time_per_start = remaining / (num_starts + 2) # +2 for buffer/final polish
    
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 5:
            break
            
        # Ensure start is valid or fix it
        s, v = _evaluate(start)
        if not v:
             # Quick fix
             for _ in range(20):
                 idx = random.randint(0, 15)
                 start[idx] = [start[idx][0] + random.randint(-5, 5), 
                               start[idx][1] + random.randint(-5, 5)]
                 s, v = _evaluate(start)
                 if v: break
             if not v:
                 continue

        # Run SA
        local_best, local_score = _simulate_annealing(start, time.time() + time_per_start, seed=100+i)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_best
            
    # Final polish: Run SA from the best found for remaining time
    if best_points is not None and time.time() < _DEADLINE - 2:
        time_left = _DEADLINE - 2 - time.time()
        if time_left > 5:
            # Use a fresh SA run from the best point found
            final_best, final_score = _simulate_annealing(best_points, time.time() + time_left, seed=999)
            if final_score > best_score:
                best_score = final_score
                best_points = final_best
    else:
        # Fallback
        best_points = _grid_start()
        best_score, _ = _evaluate(best_points)
        
    return [list(p) for p in best_points]
