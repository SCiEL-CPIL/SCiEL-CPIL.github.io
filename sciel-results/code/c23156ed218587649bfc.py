import time
import random
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return (0, 0, 0)
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    n = len(points)
    point_set = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
        
    lines = {}
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            key = _line_key(p1, p2)
            lines[key] = lines.get(key, 0) + 1
            
    score = 0
    for count in lines.values():
        if count >= 10:  # C(5,2)=10 means 5+ collinear
            return False, 0
        if count == 6:   # C(4,2)=6 means exactly 4 collinear
            score += 1
            
    return True, score

def _try_local_search(points, deadline, best_points, best_score):
    """Try all small perturbations of each point to find improvements."""
    n = len(points)
    is_valid, current_score = _evaluate(points)
    if not is_valid:
        return points, best_points, best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = points[idx]
            # Try small moves in a structured pattern
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = (nx, ny)
                    # Check collision
                    collision = False
                    for k in range(n):
                        if k != idx and points[k] == new_pt:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    points[idx] = [nx, ny]
                    is_valid, new_score = _evaluate(points)
                    if is_valid and new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in points]
                    else:
                        points[idx] = [ox, oy]
            if improved:
                break  # Restart from beginning after improvement
    
    return points, best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with a structured initial configuration
    # Use a 4x4 grid which has many 4-point lines
    current_points = [[x, y] for x in range(4) for y in range(4)]
    is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score if is_valid else 0
    
    # Phase 1: Local search from initial config
    current_points, best_points, best_score = _try_local_search(
        current_points, deadline, best_points, best_score
    )
    
    # Phase 2: Random restarts with local search
    while time.time() < deadline - 5.0:
        # Generate a new random starting point near current best
        # or completely random
        if random.random() < 0.5:
            # Perturb best solution
            start_pts = [p[:] for p in best_points]
            for i in range(3):  # Perturb 3 random points
                idx = random.randint(0, 15)
                start_pts[idx][0] += random.randint(-20, 20)
                start_pts[idx][1] += random.randint(-20, 20)
                start_pts[idx][0] = max(-1000, min(1000, start_pts[idx][0]))
                start_pts[idx][1] = max(-1000, min(1000, start_pts[idx][1]))
        else:
            # Random start
            start_pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
        
        # Check validity
        is_valid, _ = _evaluate(start_pts)
        if not is_valid:
            continue
        
        # Local search from this start
        start_pts, best_points, best_score = _try_local_search(
            start_pts, deadline, best_points, best_score
        )
        
        # If we found something better, do more local search from there
        if best_score > 0 and time.time() < deadline - 10.0:
            current_pts = [p[:] for p in best_points]
            current_pts, best_points, best_score = _try_local_search(
                current_pts, deadline, best_points, best_score
            )
    
    return best_points
