"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build map from line key to set of points on that line."""
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Compute score from line map: count lines with exactly 4 points."""
    score = 0
    for pts in line_map.values():
        if len(pts) == 4:
            score += 1
    return score

def _is_valid(line_map):
    """Check if no line has 5+ points."""
    for pts in line_map.values():
        if len(pts) >= 5:
            return False
    return True

def _evaluate(points):
    """
    Returns (is_valid, score).
    """
    line_map = _build_line_map(points)
    if not _is_valid(line_map):
        return False, 0
    return True, _score_from_line_map(line_map)

def _incremental_evaluate_move(points, line_map, idx, new_pos):
    """
    Efficiently evaluate the effect of moving point idx to new_pos.
    Returns (is_valid, new_score, new_line_map) or (False, -1, None) if invalid.
    """
    old_pos = points[idx]
    n = len(points)
    
    # Collect all lines that involve point idx
    affected_lines = set()
    for j in range(n):
        if j == idx:
            continue
        key = _line_key(old_pos, points[j])
        if key is not None:
            affected_lines.add(key)
    
    # Also check if new_pos creates any 5-collinear
    # First, build a tentative new line map
    new_line_map = {}
    for key, pts in line_map.items():
        new_pts = set(pts)
        if idx in new_pts:
            new_pts.remove(idx)
        if new_pts:
            new_line_map[key] = new_pts
    
    # Add lines from new_pos to all other points
    for j in range(n):
        if j == idx:
            continue
        key = _line_key(new_pos, points[j])
        if key is not None:
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(j)
            new_line_map[key].add(idx)
    
    # Check validity
    for key, pts in new_line_map.items():
        if len(pts) >= 5:
            return False, -1, None
    
    # Compute new score
    new_score = 0
    for key, pts in new_line_map.items():
        if len(pts) == 4:
            new_score += 1
    
    return True, new_score, new_line_map

def _local_polish(points, line_map, deadline):
    """
    Systematic local search: try moving each point to positions that
    lie on lines with exactly 3 other points, to create new 4-point lines.
    """
    current = [list(p) for p in points]
    current_line_map = {k: set(v) for k, v in line_map.items()}
    current_valid, current_score = _evaluate(current)
    if not current_valid:
        return current, 0, current_line_map
    
    best_points = [list(p) for p in current]
    best_score = current_score
    best_line_map = {k: set(v) for k, v in current_line_map.items()}
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = current[idx]
            
            # Find lines with exactly 3 points that don't include idx
            candidate_positions = set()
            for key, pts in current_line_map.items():
                if len(pts) == 3 and idx not in pts:
                    # Try to find a 4th point on this line
                    p1 = list(pts)[0]
                    p2 = list(pts)[1]
                    x1, y1 = current[p1]
                    x2, y2 = current[p2]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue
                    # Check if we can extend this line
                    # For each pair of points on the line, find integer points
                    # on the same line within bounds
                    for k in pts:
                        for l in pts:
                            if k >= l:
                                continue
                            xk, yk = current[k]
                            xl, yl = current[l]
                            ddx = xl - xk
                            ddy = yl - yk
                            if ddx == 0 and ddy == 0:
                                continue
                            g = gcd(abs(ddx), abs(ddy))
                            ddx //= g
                            ddy //= g
                            # Generate points on this line near the existing 3
                            for t in range(-3, 4):
                                nx = xk + t * ddx
                                ny = yk + t * ddy
                                if abs(nx) <= 1000 and abs(ny) <= 1000:
                                    if not any(p[0] == nx and p[1] == ny for p in current if p is not current[idx]):
                                        candidate_positions.add((nx, ny))
            
            # Also try random small perturbations
            for _ in range(20):
                nx = old_x + random.randint(-5, 5)
                ny = old_y + random.randint(-5, 5)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    if not any(p[0] == nx and p[1] == ny for p in current if p is not current[idx]):
                        candidate_positions.add((nx, ny))
            
            # Try each candidate position
            for nx, ny in candidate_positions:
                if time.time() > deadline:
                    break
                
                candidate = [list(p) for p in current]
                candidate[idx] = [nx, ny]
                
                # Quick check: does this create 5-collinear?
                valid = True
                for j in range(16):
                    if j == idx:
                        continue
                    key = _line_key((nx, ny), candidate[j])
                    if key is None:
                        continue
                    # Count how many points are on this line
                    count = 1
                    for k in range(16):
                        if k == idx or k == j:
                            continue
                        if _line_key(candidate[k], candidate[j]) == key:
                            count += 1
                    if count >= 5:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                cand_valid, cand_score = _evaluate(candidate)
                if not cand_valid:
                    continue
                
                if cand_score > current_score:
                    current = candidate
                    current_line_map = _build_line_map(current)
                    current_score = cand_score
                    improved = True
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = [list(p) for p in current]
                        best_line_map = {k: set(v) for k, v in current_line_map.items()}
                    break  # Move to next idx
    
    return best_points, best_score, best_line_map

def _simulated_annealing(points, deadline):
    """Simulated annealing to improve a starting configuration."""
    current = [list(p) for p in points]
    current_line_map = _build_line_map(current)
    current_valid, current_score = _evaluate(current)
    if not current_valid:
        return current, 0, current_line_map
    
    best_points = [list(p) for p in current]
    best_score = current_score
    best_line_map = {k: set(v) for k, v in current_line_map.items()}
    
    temp = 10.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Try a random move
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        if random.random() < 0.7:
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
        elif random.random() < 0.85:
            new_x = old_x + random.randint(-30, 30)
            new_y = old_y + random.randint(-30, 30)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        if any(p[0] == new_x and p[1] == new_y for p in current if p is not current[idx]):
            continue
        
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        
        if not cand_valid:
            cand_score = -1
        
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
            best_line_map = _build_line_map(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_line_map = _build_line_map(current)
            current_score = cand_score
        else:
            delta = cand_score - current_score
            if delta < 0:
                prob = pow(2.718281828, delta / temp)
                if random.random() < prob:
                    current = candidate
                    current_line_map = _build_line_map(current)
                    current_score = cand_score
        
        temp *= cooling
    
    return best_points, best_score, best_line_map

def _generate_initial_grid():
    """Generate a 4x4 grid of points, which has many 4-point lines."""
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(perturbation=3, scale=10):
    """Generate a perturbed 4x4 grid to break symmetry while keeping structure."""
    base = [[x, y] for x in range(4) for y in range(4)]
    points = []
    for x, y in base:
        # Add small random perturbation
        dx = random.randint(-perturbation, perturbation)
        dy = random.randint(-perturbation, perturbation)
        new_x = x * scale + dx
        new_y = y * scale + dy
        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
            points.append([new_x, new_y])
    # Ensure distinctness
    unique = []
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    while len(unique) < 16:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique

def _generate_orthogonal_grid(scale_x=10, scale_y=12, offset=0):
    """Generate a non-uniform grid that may have more 4-point lines."""
    xs = [offset + i * scale_x + random.randint(-2, 2) for i in range(4)]
    ys = [offset + i * scale_y + random.randint(-2, 2) for i in range(4)]
    points = []
    for x in xs:
        for y in ys:
            if abs(x) <= 1000 and abs(y) <= 1000:
                points.append([x, y])
    # Ensure 16 distinct points
    unique = []
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    while len(unique) < 16:
        x = random.randint(-500, 500)
        y = random.randint(-500, 500)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique

def _generate_5x5_subset():
    """Generate a random subset of a 5x5 grid."""
    # 5x5 grid has 25 points. We pick 16.
    # Using a larger scale to avoid trivial collinearities from small integers?
    # Actually, small integers are good for finding integer collinearities.
    # Let's use a 5x5 grid scaled by 10 to keep coords small but distinct.
    base = [[x * 10, y * 10] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    best_line_map = None
    
    # Strategy 1: Start with 4x4 grid
    initial_grid = _generate_initial_grid()
    grid_valid, grid_score = _evaluate(initial_grid)
    if grid_valid and grid_score > best_score:
        best_score = grid_score
        best_points = [list(p) for p in initial_grid]
        best_line_map = _build_line_map(best_points)
        
    # Strategy 2: Try 5x5 subsets
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        subset = _generate_5x5_subset()
        s_valid, s_score = _evaluate(subset)
        if s_valid and s_score > best_score:
            best_score = s_score
            best_points = [list(p) for p in subset]
            best_line_map = _build_line_map(best_points)

    # Strategy 3: Try perturbed grids
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        perturbed = _generate_perturbed_grid(perturbation=2, scale=10)
        p_valid, p_score = _evaluate(perturbed)
        if p_valid and p_score > best_score:
            best_score = p_score
            best_points = [list(p) for p in perturbed]
            best_line_map = _build_line_map(best_points)
    
    # Strategy 4: Simulated annealing from best configuration
    if best_points is None:
        best_points = _generate_initial_grid()
        best_score = _evaluate(best_points)[1]
        best_line_map = _build_line_map(best_points)
    
    # Run SA for the remaining time
    refined_points, refined_score, refined_line_map = _simulated_annealing(best_points, deadline)
    
    if refined_score > best_score:
        best_points = refined_points
        best_score = refined_score
        best_line_map = refined_line_map
    
    # Final validation
    final_valid, final_score = _evaluate(best_points)
    if not final_valid:
        # Fallback to 4x4 grid
        best_points = _generate_initial_grid()
    
    return best_points
# EVOLVE-BLOCK-END
