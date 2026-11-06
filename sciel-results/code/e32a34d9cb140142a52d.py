import time
import random
import math
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x1 * y2 - x2 * y1
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            line_counts[(A, B, C)] += 1
    
    ordinary_count = 0
    for count in line_counts.values():
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary_count += 1
                
    return ordinary_count

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    A = y2 - y1
    B = x1 - x2
    C = x1 * y2 - x2 * y1
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_ordinary_fast(points):
    """Faster count using precomputed pair keys."""
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key is not None:
                line_counts[key] += 1
    
    ordinary_count = 0
    for count in line_counts.values():
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary_count += 1
                
    return ordinary_count

def _generate_grid_config(rows, cols):
    """Generate a grid configuration."""
    points = []
    for i in range(rows):
        for j in range(cols):
            points.append([j, i])
    return points

def _generate_pappus_like():
    """Generate a Pappus-like configuration with 9 points, then extend to 12."""
    # Pappus configuration: 9 points, 9 lines of 3
    # We'll use a standard Pappus configuration
    pappus_points = [
        [0, 0], [1, 0], [2, 0],  # First line
        [0, 1], [1, 1], [2, 1],  # Second line (parallel)
        [0, 2], [1, 2], [2, 2],  # Third line (parallel)
    ]
    # This is just a 3x3 grid. For Pappus, we need specific cross-ratios.
    # Let's use a different approach: start with 3x4 grid which is known to have few ordinary lines
    
    # 3x4 grid
    points = []
    for i in range(3):
        for j in range(4):
            points.append([j, i])
    return points

def _generate_hesse_subconfig():
    """Generate a sub-configuration from Hesse configuration (9 points, 12 lines of 3)."""
    # Hesse configuration has 9 points and 12 lines of 3 points each.
    # We need 12 points, so we'll extend it.
    
    # Start with 3x4 grid as base
    points = []
    for i in range(3):
        for j in range(4):
            points.append([j, i])
    return points

def _local_search_line_aware(points, deadline, max_iter=50000):
    """
    Line-aware local search: try to move points to create new collinearities.
    """
    current = [p[:] for p in points]
    current_score = _count_ordinary_fast(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    n = len(current)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point
        idx = random.randint(0, n - 1)
        orig_x, orig_y = current[idx]
        
        # Try several candidate moves
        best_move = None
        best_move_score = current_score
        
        # Generate candidates: nearby points and points that would create collinearities
        candidates = set()
        
        # Nearby points
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = orig_x + dx, orig_y + dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    candidates.add((nx, ny))
        
        # Points that would be collinear with pairs of other points
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                x1, y1 = current[i]
                x2, y2 = current[j]
                # Line through (x1,y1) and (x2,y2)
                # Find integer points on this line near current[idx]
                dx_line = x2 - x1
                dy_line = y2 - y1
                g = math.gcd(abs(dx_line), abs(dy_line))
                if g == 0:
                    continue
                step_x = dx_line // g
                step_y = dy_line // g
                
                # Try a few points along this line
                for t in range(-3, 4):
                    if t == 0:
                        continue
                    nx = x1 + t * step_x
                    ny = y1 + t * step_y
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        # Check if this point is not already in use
                        is_used = False
                        for k in range(n):
                            if k != idx and current[k][0] == nx and current[k][1] == ny:
                                is_used = True
                                break
                        if not is_used:
                            candidates.add((nx, ny))
        
        # Evaluate a subset of candidates
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        max_eval = min(20, len(candidate_list))
        
        for cx, cy in candidate_list[:max_eval]:
            # Check distinctness
            is_distinct = True
            for i in range(n):
                if i == idx:
                    continue
                if current[i][0] == cx and current[i][1] == cy:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
            
            new_points = [p[:] for p in current]
            new_points[idx] = [cx, cy]
            
            new_score = _count_ordinary_fast(new_points)
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move = (idx, cx, cy, new_points)
        
        if best_move is not None:
            idx, cx, cy, new_points = best_move
            current = new_points
            current_score = best_move_score
            
            if current_score < best_score:
                best_score = current_score
                best = [p[:] for p in current]
                
                if best_score <= 6:
                    return best
        else:
            # No improvement found, try a random jump
            if random.random() < 0.1:
                idx = random.randint(0, n - 1)
                nx = random.randint(-10, 10)
                ny = random.randint(-10, 10)
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    is_distinct = True
                    for i in range(n):
                        if i == idx:
                            continue
                        if current[i][0] == nx and current[i][1] == ny:
                            is_distinct = False
                            break
                    if is_distinct:
                        new_points = [p[:] for p in current]
                        new_points[idx] = [nx, ny]
                        new_score = _count_ordinary_fast(new_points)
                        # Accept with some probability even if worse (simulated annealing-like)
                        if new_score <= current_score + 2:
                            current = new_points
                            current_score = new_score
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Try multiple initial configurations
    initial_configs = [
        _generate_grid_config(3, 4),  # 3x4 grid
        _generate_grid_config(2, 6),  # 2x6 grid
        _generate_grid_config(4, 3),  # 4x3 grid
        _generate_pappus_like(),
        _generate_hesse_subconfig(),
    ]
    
    # Add some random configurations
    for _ in range(5):
        pts = []
        for _ in range(12):
            pts.append([random.randint(-5, 5), random.randint(-5, 5)])
        # Ensure distinctness
        unique_pts = []
        seen = set()
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)
        if len(unique_pts) == 12:
            initial_configs.append(unique_pts)
    
    best_overall = None
    best_overall_score = float('inf')
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        # Run local search on this configuration
        result = _local_search_line_aware(config, deadline, max_iter=30000)
        
        if result is not None:
            score = _count_ordinary_fast(result)
            if score < best_overall_score:
                best_overall_score = score
                best_overall = result
                
                if best_overall_score <= 6:
                    return best_overall
    
    # If we didn't find anything good, fall back to the best initial config
    if best_overall is None:
        best_overall = initial_configs[0]
    
    return best_overall
