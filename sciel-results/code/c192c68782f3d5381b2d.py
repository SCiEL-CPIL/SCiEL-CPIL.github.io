import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def count_4point_lines(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def generate_line_candidates():
    """Generate many candidate 4-point lines with small integer coordinates."""
    candidates = []
    # Generate lines with small slopes and intercepts
    for a in range(-10, 11):
        for b in range(-10, 11):
            if a == 0 and b == 0:
                continue
            d = gcd(abs(a), abs(b))
            if d > 0:
                a //= d
                b //= d
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            
            # Try different c values to get lines within bounds
            for c in range(-50, 51):
                # Check if we can find 4 points on this line within bounds
                points_on_line = []
                # Parametrize: a*x + b*y = -c
                # If a != 0, x = (k - b*y)/a, try small y
                # If b != 0, y = (k - a*x)/b, try small x
                # Simpler: iterate x in range, solve for y
                found = 0
                for x in range(-50, 51):
                    if a * x + b * 0 == -c:  # y=0 case handled generally below
                        pass
                    # Solve a*x + b*y = -c for integer y
                    if b != 0:
                        rem = (-c - a * x) % b
                        if rem == 0:
                            y = (-c - a * x) // b
                            if abs(x) <= 1000 and abs(y) <= 1000:
                                points_on_line.append((x, y))
                                found += 1
                                if found >= 4:
                                    break
                    elif a != 0:
                        # b=0, a*x = -c
                        if (-c) % a == 0:
                            x_sol = (-c) // a
                            if abs(x_sol) <= 1000:
                                for y in range(-50, 51):
                                    points_on_line.append((x_sol, y))
                                    found += 1
                                    if found >= 4:
                                        break
                        if found >= 4:
                            break
                    else:
                        # a=0, b=0 already skipped
                        pass
                
                if found >= 4:
                    # Take 4 distinct points
                    pts = points_on_line[:4]
                    key = get_line_key(pts[0], pts[1])
                    if key is not None:
                        candidates.append((key, pts))
    
    return candidates

def greedy_line_packing(deadline, max_attempts=200):
    """Greedy approach: pick 4-point lines with minimal overlap."""
    candidates = generate_line_candidates()
    random.shuffle(candidates)
    
    best_points = None
    best_score = -1
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        selected_lines = []
        used_points = set()
        line_keys = set()
        
        for key, pts in candidates:
            if key in line_keys:
                continue
            # Check overlap
            overlap = sum(1 for p in pts if p in used_points)
            # We want minimal overlap, ideally 0 or 1
            if overlap <= 1:
                # Check if adding these points creates a 5-collinear
                new_points = list(used_points) + [p for p in pts if p not in used_points]
                if len(new_points) > 15:
                    continue
                
                # Quick check: does any line in selected_lines get a 5th point?
                valid = True
                for p in pts:
                    if p in used_points:
                        continue
                    for existing_key, existing_pts in selected_lines:
                        # Check if p lies on existing_key
                        a, b, c = existing_key
                        if a * p[0] + b * p[1] + c == 0:
                            # Count how many existing_pts are on this line + p
                            # Actually, existing_pts are all on this line by definition
                            # But we need to check if total points on this line would be 5
                            # The line has existing_pts (4 points) plus p = 5
                            # Wait, existing_pts are the 4 points we selected.
                            # If p is also on this line, then we have 5 points.
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    selected_lines.append((key, list(pts)))
                    line_keys.add(key)
                    for p in pts:
                        used_points.add(p)
                    if len(used_points) >= 16:
                        break
        
        if len(used_points) >= 16:
            points = [list(p) for p in used_points]
            score = fast_count(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    return best_points, best_score

def local_search(points, deadline, max_iter=50000):
    """Local search to improve the configuration."""
    current_points = [p[:] for p in points]
    current_score = fast_count(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 15)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_x = current_points[idx][0] + dx
        new_y = current_points[idx][1] + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_pos = [new_x, new_y]
        
        collision = False
        for i in range(16):
            if i != idx and current_points[i] == new_pos:
                collision = True
                break
        if collision:
            continue
        
        old_pos = current_points[idx][:]
        current_points[idx] = new_pos
        new_score = fast_count(current_points)
        current_points[idx] = old_pos
        
        if new_score < 0:
            continue
        
        if new_score > current_score:
            current_points[idx] = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        elif new_score == current_score and random.random() < 0.3:
            current_points[idx] = new_pos
    
    return best_points, best_score

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy 1: Greedy line packing
    gp_points, gp_score = greedy_line_packing(deadline, max_attempts=50)
    
    # Strategy 2: Grid + local search
    grid_points, grid_score = grid_configuration()
    
    # Choose better starting point
    if gp_score > grid_score:
        best_points, best_score = gp_points, gp_score
    else:
        best_points, best_score = grid_points, grid_score
    
    # Local search from best starting point
    best_points, best_score = local_search(best_points, deadline, max_iter=10000)
    
    # Try a few more greedy attempts with local search
    for _ in range(10):
        if time.time() > deadline:
            break
        gp_points, gp_score = greedy_line_packing(deadline, max_attempts=20)
        if gp_points and gp_score > best_score:
            best_points, best_score = local_search(gp_points, deadline, max_iter=5000)
    
    # Final local search
    if time.time() < deadline:
        best_points, best_score = local_search(best_points, deadline, max_iter=20000)
    
    # Ensure validity
    final_score = fast_count(best_points)
    if final_score < 0:
        best_points, best_score = grid_configuration()
    
    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy 1: Greedy line packing
    gp_points, gp_score = greedy_line_packing(deadline, max_attempts=50)
    
    # Strategy 2: Grid + local search
    grid_points, grid_score = grid_configuration()
    
    # Choose better starting point
    if gp_score > grid_score:
        best_points, best_score = gp_points, gp_score
    else:
        best_points, best_score = grid_points, grid_score
    
    # Local search from best starting point
    best_points, best_score = local_search(best_points, deadline, max_iter=10000)
    
    # Try a few more greedy attempts with local search
    for _ in range(10):
        if time.time() > deadline:
            break
        gp_points, gp_score = greedy_line_packing(deadline, max_attempts=20)
        if gp_points and gp_score > best_score:
            best_points, best_score = local_search(gp_points, deadline, max_iter=5000)
    
    # Final local search
    if time.time() < deadline:
        best_points, best_score = local_search(best_points, deadline, max_iter=20000)
    
    # Ensure validity
    final_score = fast_count(best_points)
    if final_score < 0:
        best_points, best_score = grid_configuration()
    
    return best_points
# EVOLVE-BLOCK-END
