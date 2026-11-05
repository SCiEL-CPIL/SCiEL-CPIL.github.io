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

def _build_line_table(points):
    """Build the full line count table from scratch."""
    n = len(points)
    lines = {}
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            key = _line_key(p1, p2)
            lines[key] = lines.get(key, 0) + 1
    return lines

def _score_from_lines(lines):
    """Compute score from line count table. Returns (is_valid, score)."""
    score = 0
    for count in lines.values():
        if count >= 10:
            return False, 0
        if count == 6:
            score += 1
    return True, score

def _try_local_search(points, deadline, best_points, best_score, neighborhood_radius=15):
    """Try all small perturbations of each point to find improvements using incremental line updates."""
    n = len(points)
    point_set = set(tuple(p) for p in points)
    
    # Build initial line table
    lines = _build_line_table(points)
    is_valid, current_score = _score_from_lines(lines)
    if not is_valid:
        return points, best_points, best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = points[idx]
            orig_pt = (ox, oy)
            
            # Precompute lines involving point idx (to be removed and re-added)
            old_lines = {}
            for k in range(n):
                if k == idx:
                    continue
                key = _line_key(points[idx], points[k])
                old_lines[key] = old_lines.get(key, 0) + 1
            
            # Try perturbations
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = (nx, ny)
                    if new_pt in point_set:
                        continue
                    
                    # Remove old lines involving idx
                    for key, cnt in old_lines.items():
                        lines[key] -= cnt
                        if lines[key] == 0:
                            del lines[key]
                    
                    # Add new lines involving idx
                    new_lines = {}
                    for k in range(n):
                        if k == idx:
                            continue
                        key = _line_key(new_pt, points[k])
                        new_lines[key] = new_lines.get(key, 0) + 1
                        lines[key] = lines.get(key, 0) + 1
                    
                    is_valid, new_score = _score_from_lines(lines)
                    
                    if is_valid and new_score > current_score:
                        # Accept the move
                        current_score = new_score
                        improved = True
                        points[idx] = [nx, ny]
                        point_set.remove(orig_pt)
                        point_set.add(new_pt)
                        
                        # Update old_lines to new_lines
                        old_lines = new_lines
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in points]
                        break  # Restart from beginning after improvement
                    else:
                        # Revert: remove new lines, restore old lines
                        for key, cnt in new_lines.items():
                            lines[key] -= cnt
                            if lines[key] == 0:
                                del lines[key]
                        for key, cnt in old_lines.items():
                            lines[key] = lines.get(key, 0) + cnt
                        # points[idx] unchanged, point_set unchanged
                    
                    if improved:
                        break
            
            if improved:
                break  # Restart from beginning after improvement
    
    return points, best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try multiple structured initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted
    initial_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    
    # 4x4 grid with different spacing
    initial_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    
    # 4x4 grid with affine transform
    initial_configs.append([[x * 3 + y, x - y * 2] for x in range(4) for y in range(4)])
    
    # Another affine transform
    initial_configs.append([[x * 2 + y * 3, x * 5 - y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with one row shifted
    cfg = [[x, y] for x in range(4) for y in range(4)]
    cfg[4][0] += 1
    initial_configs.append(cfg)
    
    best_points = None
    best_score = 0
    
    # Evaluate all initial configs
    for config in initial_configs:
        is_valid, score = _evaluate(config)
        if is_valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    if best_points is None:
        # Fallback to simple grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 0
    
    # Phase 1: Local search from best initial config
    current_points = [p[:] for p in best_points]
    current_points, best_points, best_score = _try_local_search(
        current_points, deadline, best_points, best_score, neighborhood_radius=15
    )
    
    # Phase 2: Local search from other good initial configs
    for config in initial_configs:
        if time.time() >= deadline - 10.0:
            break
        is_valid, score = _evaluate(config)
        if is_valid and score >= best_score - 2:
            current_points = [p[:] for p in config]
            current_points, best_points, best_score = _try_local_search(
                current_points, deadline, best_points, best_score, neighborhood_radius=15
            )
    
    # Phase 3: Random restarts with local search
    while time.time() < deadline - 5.0:
        # Generate a new random starting point
        if random.random() < 0.4:
            # Perturb best solution more aggressively
            start_pts = [p[:] for p in best_points]
            num_perturb = random.randint(2, 5)
            for _ in range(num_perturb):
                idx = random.randint(0, 15)
                start_pts[idx][0] += random.randint(-30, 30)
                start_pts[idx][1] += random.randint(-30, 30)
                start_pts[idx][0] = max(-1000, min(1000, start_pts[idx][0]))
                start_pts[idx][1] = max(-1000, min(1000, start_pts[idx][1]))
        elif random.random() < 0.7:
            # Perturb a random initial config
            base_config = random.choice(initial_configs)
            start_pts = [p[:] for p in base_config]
            for i in range(random.randint(1, 4)):
                idx = random.randint(0, 15)
                start_pts[idx][0] += random.randint(-20, 20)
                start_pts[idx][1] += random.randint(-20, 20)
                start_pts[idx][0] = max(-1000, min(1000, start_pts[idx][0]))
                start_pts[idx][1] = max(-1000, min(1000, start_pts[idx][1]))
        else:
            # Random start near origin
            start_pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
        
        # Check validity
        is_valid, _ = _evaluate(start_pts)
        if not is_valid:
            continue
        
        # Local search from this start with smaller neighborhood for speed
        radius = random.choice([8, 10, 12])
        start_pts, best_points, best_score = _try_local_search(
            start_pts, deadline, best_points, best_score, neighborhood_radius=radius
        )
        
        # If we found something better, do more local search from there
        if best_score > 0 and time.time() < deadline - 10.0:
            current_pts = [p[:] for p in best_points]
            current_pts, best_points, best_score = _try_local_search(
                current_pts, deadline, best_points, best_score, neighborhood_radius=15
            )
    
    return best_points
