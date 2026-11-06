import random
import math
import time
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize direction vector
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None # Should not happen for distinct points
    a = dy // g
    b = -dx // g
    
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        
    # Calculate c such that ax + by + c = 0
    c = -(a * x1 + b * y1)
    
    return (a, b, c)

def _evaluate(points):
    """
    Returns (score, is_valid, lines_dict).
    score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    lines_dict: dict mapping line key to set of point indices on that line.
    """
    n = len(points)
    if n != 16:
        return 0, False, {}
    
    # Check distinct and bounds
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False, {}
        if (x, y) in seen:
            return 0, False, {}
        seen.add((x, y))
        
    unique_lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                if key not in unique_lines:
                    unique_lines[key] = set()
                unique_lines[key].add(i)
                unique_lines[key].add(j)
                
    score = 0
    is_valid = True
    
    for key, pts_idx in unique_lines.items():
        count = len(pts_idx)
        if count >= 5:
            is_valid = False
            return 0, False, unique_lines
        elif count == 4:
            score += 1
            
    return score, is_valid, unique_lines

def _find_extension_points(points, line_key, current_indices):
    """
    Given a line key and indices of points on it (size 3), 
    find integer points on this line that are not occupied by any point in 'points'.
    Returns a list of candidate [x, y] positions.
    """
    a, b, c = line_key
    candidates = []
    # Search range for x and y
    # We can iterate x from -100 to 100 and check if y is integer
    # ax + by + c = 0 => by = -ax - c
    # If b != 0: y = (-ax - c) / b
    # If b == 0: ax + c = 0 => x = -c/a (single x, y can be anything? No, line is vertical x = const. y can be anything on that line? No, the line is defined by a,b,c. If b=0, a!=0, x = -c/a. y is free? No, the line is x = const. Points on the line have x = -c/a and any y. But we are looking for specific points. Actually, for a vertical line, x is fixed. We just need to find y such that (x,y) is not occupied.
    
    if b != 0:
        for x in range(-100, 101):
            val = -a * x - c
            if val % b == 0:
                y = val // b
                if -1000 <= y <= 1000:
                    # Check if occupied
                    occupied = False
                    for i, p in enumerate(points):
                        if p[0] == x and p[1] == y:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append([x, y])
    else:
        # a != 0
        val = -c
        if val % a == 0:
            x = val // a
            if -1000 <= x <= 1000:
                # y can be anything? No, the line is x = const.
                # Any y works for the line equation?
                # ax + c = 0. Yes.
                # So we need to find y such that (x,y) is not occupied.
                # We can try a few y values.
                for y in range(-100, 101):
                    if -1000 <= y <= 1000:
                        occupied = False
                        for i, p in enumerate(points):
                            if p[0] == x and p[1] == y:
                                occupied = True
                                break
                        if not occupied:
                            candidates.append([x, y])
    return candidates

def _try_local_search(points, deadline, max_steps=50000, use_sa=True):
    """Perform local search starting from given points."""
    current_score, _, lines_dict = _evaluate(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    if not best_points or best_score < 0:
        return best_points, best_score
        
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    step = 0
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        if use_sa and temperature > 0.1:
            temperature *= cooling_rate
        else:
            use_sa = False
            temperature = 0.0

        # Choose a random point to move
        idx = random.randint(0, 15)
        
        # Generate candidate new positions
        candidates = []
        
        # 1. Small perturbations
        cx, cy = points[idx]
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    occupied = False
                    for i, p in enumerate(points):
                        if i != idx and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append([nx, ny])
        
        # 2. Targeted moves: Extend 3-point lines
        # Re-compute lines_dict is expensive, but we can use the one from last eval if we stored it?
        # We stored it in the previous step's eval? No, we only stored best.
        # Let's re-evaluate to get lines_dict. This is O(N^3) roughly.
        # To save time, we can just do this every 10 steps or if we accept a move.
        if step % 10 == 0 or current_score < best_score:
            _, _, lines_dict = _evaluate(points)
        
        for key, pts_idx in lines_dict.items():
            if len(pts_idx) == 3 and idx not in pts_idx:
                ext_pts = _find_extension_points(points, key, pts_idx)
                if ext_pts:
                    # Add a few random ones to avoid bias
                    candidates.extend(random.sample(ext_pts, min(5, len(ext_pts))))
        
        # 3. Random positions
        for _ in range(5):
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            occupied = False
            for i, p in enumerate(points):
                if i != idx and p[0] == nx and p[1] == ny:
                    occupied = True
                    break
            if not occupied:
                candidates.append([nx, ny])
        
        if not candidates:
            continue
            
        new_pos = random.choice(candidates)
        new_points = [p[:] for p in points]
        new_points[idx] = new_pos
        
        new_score, is_valid, _ = _evaluate(new_points)
        
        if is_valid:
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            elif new_score == current_score:
                if random.random() < 0.5:
                    points = new_points
            else:
                # Simulated Annealing: accept worse moves
                if use_sa and temperature > 0:
                    delta = current_score - new_score
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        points = new_points
                        current_score = new_score
        # If invalid, reject

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    initial_configs = []
    
    # 1. 4x4 Grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. Sheared Grid
    sheared = []
    for y in range(4):
        for x in range(4):
            sheared.append([x + y, y])
    initial_configs.append(sheared)
    
    # 3. Random Cluster
    random.seed(42)
    pts = []
    for _ in range(16):
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if [x, y] not in pts:
            pts.append([x, y])
    while len(pts) < 16:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if [x, y] not in pts:
            pts.append([x, y])
    initial_configs.append(pts)
    
    best_score = 0
    best_points = None
    
    for config in initial_configs:
        if time.time() >= deadline - 10:
            break
            
        # Evaluate initial
        score, valid, _ = _evaluate(config)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run local search
        if time.time() < deadline - 5:
            search_deadline = min(deadline - 2, time.time() + 30)
            improved_points, improved_score = _try_local_search(config, search_deadline, max_steps=10000, use_sa=True)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Final polish on the best result
    if best_points and time.time() < deadline - 2:
        final_deadline = deadline - 1
        improved_points, improved_score = _try_local_search(best_points, final_deadline, max_steps=50000, use_sa=True)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
            
    return best_points

# EVOLVE-BLOCK-END
