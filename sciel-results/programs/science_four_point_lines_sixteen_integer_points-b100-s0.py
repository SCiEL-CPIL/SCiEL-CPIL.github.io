import random
import time
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _build_line_index(points):
    """Build mapping from line_key -> set of point indices on that line."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    """Count lines with exactly 4 points, check for 5+. Returns (valid, score)."""
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_update(points, line_points, old_idx, old_pt, new_pt):
    """
    Update line_points after moving point old_idx from old_pt to new_pt.
    Returns updated line_points (mutates in place for speed, but we copy structure).
    """
    n = len(points)
    # Remove old point from all lines it was on
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    # Add new point to lines it's on
    for j in range(n):
        if j == old_idx:
            continue
        key = _get_line_key(new_pt, points[j])
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(old_idx)
        line_points[key].add(j)
    
    return line_points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill
    
    # Initial: 4x4 grid - has the affine plane of order 4 structure
    # 20 lines of exactly 4 points each
    points = [[x, y] for x in range(4) for y in range(4)]
    
    line_points = _build_line_index(points)
    is_valid, current_score = _evaluate_from_index(points, line_points)
    
    best_points = [p[:] for p in points]
    best_score = current_score
    best_line_points = {k: set(v) for k, v in line_points.items()}
    
    if not is_valid:
        return best_points
    
    random.seed(42)
    
    # Local search with incremental evaluation
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        old_pt = [old_x, old_y]
        
        # Generate candidate moves
        mode = random.random()
        if mode < 0.6:
            # Small perturbation - stay close to preserve structure
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        elif mode < 0.85:
            # Medium jump
            new_x = random.randint(-5, 5)
            new_y = random.randint(-5, 5)
        else:
            # Larger jump for exploration
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        is_distinct = True
        for i, p in enumerate(points):
            if i != idx:
                if p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
        if not is_distinct:
            continue
        
        # Apply move tentatively
        points[idx] = new_point
        
        # Incremental update
        new_line_points = _incremental_update(points, line_points, idx, old_pt, new_point)
        
        # Evaluate
        is_valid, new_score = _evaluate_from_index(points, new_line_points)
        
        # Acceptance criteria
        accept = False
        if is_valid and new_score > current_score:
            accept = True
        elif is_valid and new_score == current_score and random.random() < 0.15:
            # Plateau move to escape local optima
            accept = True
        elif not is_valid and new_score > current_score and random.random() < 0.05:
            # Rarely accept invalid if it has higher potential score
            accept = False  # Too risky, stick to valid only
        
        if accept:
            current_score = new_score
            line_points = new_line_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                best_line_points = {k: set(v) for k, v in new_line_points.items()}
        else:
            # Revert
            points[idx] = old_pt
            # Rebuild line_points from scratch to ensure consistency
            line_points = _build_line_index(points)
            # Re-evaluate to sync state
            is_valid, current_score = _evaluate_from_index(points, line_points)
    
    return best_points
