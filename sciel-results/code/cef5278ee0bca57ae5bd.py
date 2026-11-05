import time
import random
from math import gcd
from itertools import combinations

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

def _generate_pg23_candidates():
    """
    Generate candidate configurations inspired by PG(2,3).
    PG(2,3) has 16 points and 21 lines of 4 points.
    We try to find integer embeddings that preserve many of these 4-point lines.
    
    Strategy: Use affine transformations and small perturbations of structured grids.
    """
    candidates = []
    
    # Base 4x4 grid (score ~10)
    base_grid = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    candidates.append([p[:] for p in base_grid])
    
    # Try various affine-like transformations
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    if a * d - b * c == 0:
                        continue
                    pts = []
                    for x in range(-1, 3):
                        for y in range(-1, 3):
                            nx = a * x + b * y
                            ny = c * x + d * y
                            if abs(nx) <= 1000 and abs(ny) <= 1000:
                                pts.append([nx, ny])
                    if len(pts) == 16 and len(set(tuple(p) for p in pts)) == 16:
                        candidates.append(pts)
    
    # Try structured patterns with different spacings
    for sx in [2, 3, 4, 5]:
        for sy in [2, 3, 4, 5]:
            pts = [[x * sx, y * sy] for x in range(-1, 3) for y in range(-1, 3)]
            if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                candidates.append(pts)
    
    # Try diagonal grids
    for shift in range(-2, 3):
        pts = []
        for x in range(-1, 3):
            for y in range(-1, 3):
                pts.append([x + shift, y - shift])
        candidates.append(pts)
    
    return candidates

def _local_search(points, time_limit=5.0):
    """
    Perform local search starting from given points.
    Uses simulated annealing-like acceptance to escape local optima.
    """
    start = time.time()
    is_valid, current_score = _evaluate(points)
    if not is_valid:
        return points, 0
    
    best_points = [p[:] for p in points]
    best_score = current_score
    current_points = [p[:] for p in points]
    
    temperature = 2.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        prev_points = [p[:] for p in current_points]
        
        # Choose a move type
        r = random.random()
        if r < 0.5:
            # Move one point slightly
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            step = random.randint(1, 8)
            angle = random.choice([0, 1, -1, 2, -2])  # 0, 45, -45, 90, -90 degrees roughly
            if angle == 0:
                nx, ny = ox + step, oy
            elif angle == 1:
                nx, ny = ox + step, oy + step
            elif angle == -1:
                nx, ny = ox + step, oy - step
            elif angle == 2:
                nx, ny = ox, oy + step
            else:
                nx, ny = ox, oy - step
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
        elif r < 0.7:
            # Move one point more
            idx = random.randint(0, 15)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            current_points[idx] = [nx, ny]
        elif r < 0.85:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
        else:
            # Move two points
            i, j = random.sample(range(16), 2)
            current_points[i][0] += random.randint(-5, 5)
            current_points[i][1] += random.randint(-5, 5)
            current_points[j][0] += random.randint(-5, 5)
            current_points[j][1] += random.randint(-5, 5)
            # Clamp
            for k in (i, j):
                current_points[k][0] = max(-1000, min(1000, current_points[k][0]))
                current_points[k][1] = max(-1000, min(1000, current_points[k][1]))
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or random.random() < pow(2.718281828, delta / temperature):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points = prev_points
        else:
            current_points = prev_points
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate diverse initial candidates
    candidates = _generate_pg23_candidates()
    
    best_points = None
    best_score = 0
    
    # Evaluate each candidate and run local search on the best ones
    for i, candidate in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        is_valid, score = _evaluate(candidate)
        if not is_valid:
            continue
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in candidate]
        
        # Run local search on promising candidates
        if score >= 8 and time.time() < deadline - 15:
            local_best, local_score = _local_search(candidate, time_limit=min(5.0, max(1.0, deadline - time.time() - 10)))
            if local_score > best_score:
                best_score = local_score
                best_points = local_best
    
    # If we still have time, do more random restarts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_points is not None:
        while time.time() < deadline - 5:
            # Start from best found and perturb
            start_pts = [p[:] for p in best_points]
            for _ in range(5):
                idx = random.randint(0, 15)
                start_pts[idx][0] += random.randint(-10, 10)
                start_pts[idx][1] += random.randint(-10, 10)
                start_pts[idx][0] = max(-1000, min(1000, start_pts[idx][0]))
                start_pts[idx][1] = max(-1000, min(1000, start_pts[idx][1]))
            
            local_best, local_score = _local_search(start_pts, time_limit=min(3.0, max(0.5, deadline - time.time() - 5)))
            if local_score > best_score:
                best_score = local_score
                best_points = local_best
    
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    
    return best_points
