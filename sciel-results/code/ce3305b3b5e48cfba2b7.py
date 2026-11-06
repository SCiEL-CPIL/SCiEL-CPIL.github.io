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
    
    if dx == 0 and dy == 0:
        return None
        
    g = _gcd(abs(dx), abs(dy))
    a = dy // g
    b = -dx // g
    
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        
    c = -(a * x1 + b * y1)
    
    return (a, b, c)

def _evaluate(points):
    n = len(points)
    
    # Map from line key to set of point indices on that line
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
                
    score = 0
    is_valid = True
    
    for key, pts in line_map.items():
        if len(pts) >= 5:
            is_valid = False
            return 0, False
        elif len(pts) == 4:
            score += 1
            
    return score, True

def _evaluate_fast(points):
    """
    Faster evaluation using bitmasks for small n.
    Returns (score, is_valid).
    """
    n = len(points)
    
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                if key not in line_map:
                    line_map[key] = 0
                line_map[key] |= (1 << i) | (1 << j)
                
    score = 0
    is_valid = True
    
    for key, mask in line_map.items():
        cnt = bin(mask).count('1')
        if cnt >= 5:
            is_valid = False
            return 0, False
        elif cnt == 4:
            score += 1
            
    return score, True

def _try_local_search(points, deadline, max_steps=50000, temp=0.5):
    """
    Perform simulated annealing local search.
    """
    current_score, _ = _evaluate_fast(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    step = 0
    start_time = time.time()
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Choose a random point to move
        idx = random.randint(0, 15)
        
        # Generate candidate new positions
        candidates = []
        
        cx, cy = points[idx]
        
        # 1. Small perturbations
        for dx in range(-2, 3):
            for dy in range(-2, 3):
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
        
        # 2. Random positions
        for _ in range(3):
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            occupied = False
            for i, p in enumerate(points):
                if i != idx and p[0] == nx and p[1] == ny:
                    occupied = True
                    break
            if not occupied:
                candidates.append([nx, ny])
        
        # 3. Smart moves: align with existing 3-point lines
        # Build line map
        line_map = {}
        for i, j in pairs:
            key = _line_key(points[i], points[j])
            if key:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        
        for key, pts_on_line in line_map.items():
            if len(pts_on_line) == 3 and idx not in pts_on_line:
                a, b, c = key
                # Find integer points on this line
                # ax + by = -c
                # We need small integer solutions
                if a != 0:
                    # x = (-c - by) / a
                    # Try small y values
                    for y in range(-15, 16):
                        val = -c - b * y
                        if val % a == 0:
                            x = val // a
                            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                # Check if this point is free
                                if (x, y) not in [(points[i][0], points[i][1]) for i in pts_on_line]:
                                    occupied = False
                                    for i, p in enumerate(points):
                                        if i != idx and p[0] == x and p[1] == y:
                                            occupied = True
                                            break
                                    if not occupied:
                                        candidates.append([x, y])
                elif b != 0:
                    for x in range(-15, 16):
                        val = -c - a * x
                        if val % b == 0:
                            y = val // b
                            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                if (x, y) not in [(points[i][0], points[i][1]) for i in pts_on_line]:
                                    occupied = False
                                    for i, p in enumerate(points):
                                        if i != idx and p[0] == x and p[1] == y:
                                            occupied = True
                                            break
                                    if not occupied:
                                        candidates.append([x, y])
        
        if not candidates:
            continue
            
        new_pos = random.choice(candidates)
        new_points = [p[:] for p in points]
        new_points[idx] = new_pos
        
        new_score, is_valid = _evaluate_fast(new_points)
        
        if not is_valid:
            continue
            
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            # Simulated annealing: accept worse moves with some probability
            delta = new_score - current_score
            # Temperature cools over time
            elapsed = time.time() - start_time
            total_time = deadline - start_time
            if total_time > 0:
                progress = min(1.0, elapsed / total_time)
                t = temp * (1.0 - progress)
            else:
                t = temp
            
            if delta >= 0:
                points = new_points
                current_score = new_score
            else:
                # Accept with probability exp(delta / t)
                if t > 0:
                    prob = math.exp(delta / t)
                    if random.random() < prob:
                        points = new_points
                        current_score = new_score
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    # Initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    # 4x4 grid shifted
    initial_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    # 4x4 grid scaled
    initial_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    # 4x4 grid with different spacing
    initial_configs.append([[x * 3, y * 3] for x in range(4) for y in range(4)])
    # 4x4 grid centered
    initial_configs.append([[x - 1, y - 1] for x in range(4) for y in range(4)])
    
    # Random configurations
    for _ in range(10):
        pts = []
        while len(pts) < 16:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        initial_configs.append(pts)
    
    best_score = 0
    best_points = None
    
    # Evaluate each initial config and run local search
    for config in initial_configs:
        if time.time() >= deadline - 30:
            break
            
        score, _ = _evaluate_fast(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run local search
        if time.time() < deadline - 20:
            search_deadline = min(deadline - 10, time.time() + 15)
            improved_points, improved_score = _try_local_search(config, search_deadline, max_steps=30000, temp=1.0)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Final polish on the best result
    if best_points and time.time() < deadline - 5:
        final_deadline = deadline - 2
        improved_points, improved_score = _try_local_search(best_points, final_deadline, max_steps=50000, temp=0.5)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    return best_points

# EVOLVE-BLOCK-END
