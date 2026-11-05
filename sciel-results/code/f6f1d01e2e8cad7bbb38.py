import time
import random
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

def _evaluate_from_index(line_points):
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            return -1, score  # Invalid
        elif cnt == 4:
            score += 1
    return 0, score  # Valid, score

def _incremental_update(points, line_points, old_idx, new_pt):
    n = len(points)
    # Remove old_idx from all lines it was on
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    # Add new point to lines defined with other points
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Initial: 4x4 grid (affine plane of order 4) - 20 four-point lines
    points = [[x, y] for x in range(4) for y in range(4)]
    
    line_points = _build_line_index(points)
    valid_flag, current_score = _evaluate_from_index(line_points)
    
    best_points = [p[:] for p in points]
    best_score = current_score
    
    if valid_flag != 0:
        return best_points
    
    random.seed(42)
    
    no_improve = 0
    max_no_improve = 80
    plateau_prob = 0.2  # Decays over time
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        old_pt = [old_x, old_y]
        
        # Multi-scale move generation
        mode = random.random()
        if mode < 0.5:
            # Small perturbation
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        elif mode < 0.8:
            # Medium jump
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
        else:
            # Larger jump for exploration
            new_x = old_x + random.randint(-25, 25)
            new_y = old_y + random.randint(-25, 25)
        
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
        new_line_points = _incremental_update(points, line_points, idx, new_point)
        
        # Evaluate
        new_valid_flag, new_score = _evaluate_from_index(new_line_points)
        
        # Acceptance criteria
        accept = False
        if new_valid_flag == 0:  # Valid
            if new_score > current_score:
                accept = True
            elif new_score == current_score and random.random() < plateau_prob:
                accept = True
        # Reject invalid moves (new_valid_flag != 0)
        
        if accept:
            current_score = new_score
            line_points = new_line_points
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = old_pt
            # Rebuild to ensure consistency
            line_points = _build_line_index(points)
            valid_flag, current_score = _evaluate_from_index(line_points)
            no_improve += 1
        
        # Decay plateau probability slowly
        plateau_prob = max(0.05, plateau_prob * 0.999)
        
        # Restart with perturbation if stuck
        if no_improve >= max_no_improve:
            no_improve = 0
            # Apply 2-3 random larger moves
            for _ in range(random.randint(2, 3)):
                ridx = random.randint(0, 15)
                rx = random.randint(-30, 30)
                ry = random.randint(-30, 30)
                if abs(rx) <= 1000 and abs(ry) <= 1000:
                    # Check distinctness
                    distinct = True
                    for i, p in enumerate(points):
                        if i != ridx and p[0] == rx and p[1] == ry:
                            distinct = False
                            break
                    if distinct:
                        points[ridx] = [rx, ry]
            
            # Rebuild and re-evaluate
            line_points = _build_line_index(points)
            valid_flag, current_score = _evaluate_from_index(line_points)
            
            # If invalid after perturbation, revert to best
            if valid_flag != 0:
                points = [p[:] for p in best_points]
                line_points = _build_line_index(points)
                valid_flag, current_score = _evaluate_from_index(line_points)
        
        # Safety check
        if time.time() > deadline - 5:
            break
    
    return best_points
