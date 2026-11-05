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
    if n != 16:
        return False, 0
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
    # Pre-compute pairs to avoid repeated indexing in inner loop? 
    # With N=16, nested loops are fine.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue # Should not happen if distinct
            g = gcd(abs(dx), abs(dy))
            ndx = dx // g
            ndy = dy // g
            if ndx < 0 or (ndx == 0 and ndy < 0):
                ndx = -ndx
                ndy = -ndy
            c = ndy * x1 - ndx * y1
            key = (ndx, ndy, c)
            lines[key] = lines.get(key, 0) + 1
            
    score = 0
    for count in lines.values():
        if count >= 10:  # C(5,2)=10 means 5+ collinear
            return False, 0
        if count == 6:   # C(4,2)=6 means exactly 4 collinear
            score += 1
            
    return True, score

def _generate_initial_points():
    """Generate a 4x4 grid centered at 0."""
    return [[x, y] for x in range(-1, 3) for y in range(-1, 3)]

def _perturb(points, intensity=5):
    """Create a perturbed copy of the points."""
    new_pts = [p[:] for p in points]
    for i in range(16):
        if random.random() < 0.5:
            new_pts[i][0] += random.randint(-intensity, intensity)
            new_pts[i][1] += random.randint(-intensity, intensity)
            # Clamp
            new_pts[i][0] = max(-1000, min(1000, new_pts[i][0]))
            new_pts[i][1] = max(-1000, min(1000, new_pts[i][1]))
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    current_points = _generate_initial_points()
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        # Fallback to simple grid if initial is somehow invalid (shouldn't be)
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # If we already have a high score, we can be more conservative
    # But 10 is the grid score. We hope to beat it.
    
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Move one point slightly
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            # Small step
            nx = ox + random.randint(-5, 5)
            ny = oy + random.randint(-5, 5)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
        elif move_type < 0.85:
            # Move one point significantly
            idx = random.randint(0, 15)
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            current_points[idx] = [nx, ny]
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                no_improve_steps = 0
            elif new_score == best_score:
                no_improve_steps += 1
            else:
                # Revert if worse, unless we want to explore
                # Simple hill climbing: revert
                # We need to revert the move. 
                # It's easier to keep a copy of the previous state or revert manually.
                # Since we didn't save the previous state explicitly for all move types,
                # let's just revert by re-evaluating? No, that's slow.
                # Let's store the previous state.
                pass # This logic is flawed if we don't store prev state.
        else:
            # Invalid, revert
            pass

    # The above loop is inefficient because reverting is not handled cleanly.
    # Let's rewrite the search loop properly.
    
    return best_points

# Rewritten construct with proper state management

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial configuration
    current_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    while time.time() < deadline:
        prev_points = [p[:] for p in current_points]
        
        # Random move
        r = random.random()
        if r < 0.6:
            # Small move
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            nx = ox + random.randint(-5, 5)
            ny = oy + random.randint(-5, 5)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
        elif r < 0.8:
            # Medium move
            idx = random.randint(0, 15)
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            current_points[idx] = [nx, ny]
        else:
            # Swap
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            elif new_score == current_score:
                # Accept equal score to allow random walk
                pass
            else:
                # Revert
                current_points = prev_points
        else:
            # Revert
            current_points = prev_points
            
    return best_points
