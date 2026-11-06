import time
import random
import math
from itertools import combinations

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    
    lines_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            count = 0
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    count += 1
                    indices_on_line.add(k)
            if count >= 5:
                return -1
            if count == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
    return len(lines_found)

def _score_with_penalty(points):
    """
    Score = (# of 4-point lines) - 100 * (# of 5+ collinear lines)
    This allows the optimizer to navigate through invalid states.
    """
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -10000
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -10000
    
    lines_4 = 0
    lines_5_plus = 0
    seen_lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices_on_line.add(k)
            if len(indices_on_line) >= 2:
                line_id = tuple(sorted(indices_on_line))
                if line_id not in seen_lines:
                    seen_lines.add(line_id)
                    if len(indices_on_line) == 4:
                        lines_4 += 1
                    elif len(indices_on_line) >= 5:
                        lines_5_plus += 1
    
    return lines_4 - 100 * lines_5_plus

def _make_5x5_grid_subset():
    """Create a 16-point subset from a 5x5 grid, centered."""
    grid = [[x - 2, y - 2] for x in range(5) for y in range(5)]
    random.shuffle(grid)
    return [p[:] for p in grid[:16]]

def _make_6x3_grid():
    """Create a 6x3 grid subset (18 points, pick 16)."""
    grid = [[x - 2, y - 1] for x in range(6) for y in range(3)]
    random.shuffle(grid)
    return [p[:] for p in grid[:16]]

def _make_random_dense():
    """Random points in a small dense region."""
    points = []
    for _ in range(16):
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        points.append([x, y])
    # Ensure distinct
    unique = []
    seen = set()
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    while len(unique) < 16:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            unique.append([x, y])
    return unique[:16]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 4x4 grid (baseline)
    grid44 = [[x - 2, y - 2] for x in range(4) for y in range(4)]
    initial_configs.append([p[:] for p in grid44])
    
    # 5x5 grid subsets
    for _ in range(3):
        initial_configs.append(_make_5x5_grid_subset())
    
    # 6x3 grid
    initial_configs.append(_make_6x3_grid())
    
    # Random dense
    for _ in range(2):
        initial_configs.append(_make_random_dense())
    
    best_points = None
    best_score = -10000
    
    # Evaluate all initial configs
    for config in initial_configs:
        score = _score_with_penalty(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    max_iterations = 50000
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        no_improve += 1
        
        # Choose move type: single point move or pair swap
        if random.random() < 0.3:
            # Pair swap
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            
            # Try swapping
            current_points[i] = old_j[:]
            current_points[j] = old_i[:]
            
            new_score = _score_with_penalty(current_points)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept with probability exp((new - current)/T)
                delta = new_score - current_score
                if delta < 0 and random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_points[i] = old_i[:]
                    current_points[j] = old_j[:]
        else:
            # Single point move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Step size based on temperature
            if T > 1.0:
                step = random.randint(-8, 8)
                step2 = random.randint(-8, 8)
            else:
                step = random.randint(-3, 3)
                step2 = random.randint(-3, 3)
            
            new_x = max(-1000, min(1000, old_x + step))
            new_y = max(-1000, min(1000, old_y + step2))
            
            if new_x == old_x and new_y == old_y:
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = _score_with_penalty(current_points)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                delta = new_score - current_score
                if delta < 0 and random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[idx] = [old_x, old_y]
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Restart if stuck too long
        if no_improve > max_no_improve:
            # Restart from best or a new random config
            if random.random() < 0.5:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_points = _make_5x5_grid_subset()
                current_score = _score_with_penalty(current_points)
            no_improve = 0
            T = 5.0  # Reheat
    
    # Final validation: ensure no 5 collinear
    final_score = _count_lines_exactly_4(best_points)
    if final_score < 0:
        # Find a valid config from best_points neighborhood
        # Try small perturbations
        for _ in range(1000):
            trial = [p[:] for p in best_points]
            idx = random.randint(0, 15)
            trial[idx][0] += random.randint(-2, 2)
            trial[idx][1] += random.randint(-2, 2)
            trial[idx][0] = max(-1000, min(1000, trial[idx][0]))
            trial[idx][1] = max(-1000, min(1000, trial[idx][1]))
            if _count_lines_exactly_4(trial) >= 0:
                best_points = trial
                break
    
    return best_points
