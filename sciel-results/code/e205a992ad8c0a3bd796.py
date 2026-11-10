"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0 and b < 0:
        b = -b
        c = -c
    return (a, b, c)

def _build_line_map(points):
    """Build line map: line_key -> set of point indices."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points, and check no 5+ collinear."""
    count_4 = 0
    max_collinear = 0
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 5:
            return -1, sz  # invalid
        if sz == 4:
            count_4 += 1
        if sz > max_collinear:
            max_collinear = sz
    return count_4, max_collinear

def _count_4_points(points):
    """Count lines containing exactly 4 of the given points. Returns -1 if invalid."""
    line_map = _build_line_map(points)
    count_4, _ = _score_from_line_map(line_map)
    return count_4

def _is_valid(points):
    """Check validity: 16 distinct, bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return False
    return True

def _simulated_annealing(points, start_temp=10.0, end_temp=0.01, max_time=50.0, seed=None):
    """Simulated annealing to maximize 4-point lines."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = -100
    
    best = [list(p) for p in current]
    best_score = cur_score if cur_score >= 0 else 0
    if cur_score < 0:
        best_score = 0
        best = [list(p) for p in current]
    
    temp = start_temp
    total_time = max_time
    steps = 0
    stuck_counter = 0
    
    while time.time() - start_time < max_time:
        temp = max(end_temp, start_temp * (1 - (time.time() - start_time) / total_time))
        
        idx = random.randint(0, 15)
        r = random.random()
        if r < 0.5:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        elif r < 0.8:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
        else:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
        
        if dx == 0 and dy == 0:
            continue
        
        new_x = current[idx][0] + dx
        new_y = current[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_points = [list(p) for p in current]
        new_points[idx] = [new_x, new_y]
        
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        
        new_line_map = _build_line_map(new_points)
        new_score, _ = _score_from_line_map(new_line_map)
        if new_score < 0:
            new_score = -100
        
        delta = new_score - cur_score
        if delta >= 0:
            current = new_points
            cur_score = new_score
            stuck_counter = 0
            if cur_score > best_score:
                best = [list(p) for p in current]
                best_score = cur_score
        else:
            if temp > 0:
                prob = min(1.0, max(0.0, delta / temp))
                if random.random() < prob:
                    current = new_points
                    cur_score = new_score
            stuck_counter += 1
            
            # Diversification: if stuck for a while, make a large random jump
            if stuck_counter > 2000:
                stuck_counter = 0
                jump_idx = random.randint(0, 15)
                jump_dx = random.randint(-100, 100)
                jump_dy = random.randint(-100, 100)
                nx = current[jump_idx][0] + jump_dx
                ny = current[jump_idx][1] + jump_dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    current[jump_idx] = [nx, ny]
                    if len(set(tuple(p) for p in current)) == 16:
                        line_map = _build_line_map(current)
                        cur_score, _ = _score_from_line_map(line_map)
                        if cur_score < 0:
                            cur_score = -100
                        if cur_score > best_score:
                            best = [list(p) for p in current]
                            best_score = cur_score
        steps += 1
    
    return best, best_score

def _generate_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_random_points():
    pts = []
    seen = set()
    while len(pts) < 16:
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts

def construct():
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin before 150s limit
    
    random.seed(12345)
    
    best_points = None
    best_score = -1
    
    # 1. Try 4x4 grid with SA
    initial = _generate_grid_4x4()
    pts, score = _simulated_annealing(initial, start_temp=10.0, end_temp=0.01, 
                                       max_time=30, seed=42)
    if score > best_score:
        best_points = pts
        best_score = score
        
    # 2. Try random starts with SA
    for trial in range(5):
        if time.time() - start_time > deadline - 10:
            break
        initial = _generate_random_points()
        time_per_trial = min(20, (deadline - 10 - (time.time() - start_time)) / (5 - trial))
        pts, score = _simulated_annealing(initial, start_temp=20.0, end_temp=0.01, 
                                           max_time=time_per_trial, seed=100 + trial)
        if score > best_score:
            best_points = pts
            best_score = score
            
    # 3. Try perturbed grid
    for trial in range(3):
        if time.time() - start_time > deadline - 5:
            break
        initial = _generate_grid_4x4()
        # Perturb
        for i in range(16):
            if random.random() < 0.5:
                initial[i][0] += random.randint(-3, 3)
                initial[i][1] += random.randint(-3, 3)
        time_per_trial = min(15, (deadline - 5 - (time.time() - start_time)) / (3 - trial))
        pts, score = _simulated_annealing(initial, start_temp=10.0, end_temp=0.01, 
                                           max_time=time_per_trial, seed=200 + trial)
        if score > best_score:
            best_points = pts
            best_score = score

    # Final validation
    if best_points is None:
        return _generate_grid_4x4()
    
    if not _is_valid(best_points):
        return _generate_grid_4x4()
        
    return best_points
# EVOLVE-BLOCK-END
