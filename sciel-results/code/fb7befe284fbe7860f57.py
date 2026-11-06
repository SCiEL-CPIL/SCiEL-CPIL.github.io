"""Bootstrap spawn: model-and-optimize for 16 integer points, maximize exact-4 lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx_n, dy_n = dx // g, dy // g
    # Normalize direction
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    """
    Returns (num_exact_4_lines, has_5_collinear, line_dict).
    points: list of 16 [x, y]
    """
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5, line_pts

def _get_candidates_on_line(line_key, points, exclude_indices):
    """
    Find integer points on the line within bounds, excluding those at exclude_indices.
    """
    a, b, c = line_key
    # Direction vector (b, -a)
    dx_step = b
    dy_step = -a
    
    if not exclude_indices:
        return []
    
    # Use one point on the line as reference
    ref_idx = list(exclude_indices)[0]
    x_ref, y_ref = points[ref_idx]
    
    candidates = []
    existing_set = {tuple(points[i]) for i in range(len(points))}
    
    # Search range for k
    # x = x_ref + k*dx_step, y = y_ref + k*dy_step
    # |x| <= 1000, |y| <= 1000
    
    k_min = -2000
    k_max = 2000
    
    # Refine bounds
    if dx_step > 0:
        k_min = max(k_min, math.ceil((-1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((1000 - x_ref) / dx_step))
    elif dx_step < 0:
        k_min = max(k_min, math.ceil((1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((-1000 - x_ref) / dx_step))
    
    if dy_step > 0:
        k_min = max(k_min, math.ceil((-1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((1000 - y_ref) / dy_step))
    elif dy_step < 0:
        k_min = max(k_min, math.ceil((1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((-1000 - y_ref) / dy_step))
        
    if k_min > k_max:
        return []
        
    for k in range(k_min, k_max + 1):
        x = x_ref + k * dx_step
        y = y_ref + k * dy_step
        pt = (x, y)
        if pt not in existing_set:
            # Check if this point is one of the exclude_indices (it shouldn't be if we built existing_set correctly, 
            # but we want to replace points NOT on the line, or maybe we are adding to a line?
            # This function is used to find where to PLACE a point.
            # The caller decides which index to replace.
            candidates.append(pt)
            
    return candidates

def _repair(points, rng):
    """
    Attempt to fix 5-collinear violations by moving points.
    Returns new points, or None if failed.
    """
    score, has_5, line_pts = _score_lines(points)
    if not has_5:
        return points
    
    # Find a line with >= 5 points
    bad_lines = [k for k, v in line_pts.items() if len(v) >= 5]
    if not bad_lines:
        return points
        
    # Pick a random bad line
    bad_line = rng.choice(bad_lines)
    pts_on_line = line_pts[bad_line]
    
    # Move one point from this line to a random valid location
    # Choose a point on the line that is NOT in a critical 4-line (hard to check, just random)
    idx_to_move = rng.choice(list(pts_on_line))
    
    # Try random moves until valid
    for _ in range(50):
        old_pt = points[idx_to_move][:]
        new_x = rng.randint(-1000, 1000)
        new_y = rng.randint(-1000, 1000)
        
        # Check distinctness
        if any([new_x, new_y] == points[i] for i in range(16) if i != idx_to_move):
            continue
            
        points[idx_to_move] = [new_x, new_y]
        
        new_score, new_has_5, _ = _score_lines(points)
        if not new_has_5:
            return points
        # If still has 5, keep trying (maybe this move created another or didn't fix)
        # To avoid infinite loop, we just return the best we found or original?
        # For GA, we want valid offspring. If we can't fix it quickly, discard.
        
    # Failed to fix quickly
    return None

def _mutate_complete(points, rng):
    """
    Mutation: Try to complete a 3-point line to a 4-point line.
    """
    score, has_5, line_pts = _score_lines(points)
    if has_5:
        return points
        
    # Find lines with exactly 3 points
    three_lines = [k for k, v in line_pts.items() if len(v) == 3]
    if not three_lines:
        return points
        
    # Pick a random 3-line
    target_line = rng.choice(three_lines)
    pts_on_line = line_pts[target_line]
    
    # Find candidates on this line
    candidates = _get_candidates_on_line(target_line, points, pts_on_line)
    if not candidates:
        return points
        
    # Pick a candidate
    new_pt = rng.choice(candidates)
    
    # Pick a point to replace (one NOT on this line, to maintain the 3 base points? 
    # Actually, if we replace a point ON the line, we might break it.
    # We should replace a point OFF the line, or swap.
    # If we replace a point OFF the line, we add a 4th point.
    off_line_indices = [i for i in range(16) if i not in pts_on_line]
    if not off_line_indices:
        return points
        
    idx_to_replace = rng.choice(off_line_indices)
    
    old_pt = points[idx_to_replace][:]
    points[idx_to_replace] = list(new_pt)
    
    # Check validity
    new_score, new_has_5, _ = _score_lines(points)
    if not new_has_5:
        # Accept if score improves or stays same (neutral) to allow exploration?
        # Usually we want improvement.
        if new_score >= score:
            return points
        else:
            points[idx_to_replace] = old_pt
            return points
    else:
        points[idx_to_replace] = old_pt
        return points

def _mutate_perturb(points, rng):
    """
    Mutation: Small random perturbation of a random point.
    """
    idx = rng.randint(0, 15)
    old_pt = points[idx][:]
    step = rng.randint(1, 10)
    new_x = old_pt[0] + rng.randint(-step, step)
    new_y = old_pt[1] + rng.randint(-step, step)
    
    if abs(new_x) > 1000 or abs(new_y) > 1000:
        return points
    if any([new_x, new_y] == points[i] for i in range(16) if i != idx):
        return points
        
    points[idx] = [new_x, new_y]
    
    # Check validity
    _, has_5, _ = _score_lines(points)
    if has_5:
        points[idx] = old_pt
    return points

def _crossover(p1, p2, rng):
    """
    Uniform crossover: pick each point from either parent.
    """
    child = []
    for i in range(16):
        if rng.random() < 0.5:
            child.append(p1[i][:])
        else:
            child.append(p2[i][:])
    return child

def _initialize_population(rng):
    """
    Create diverse initial population.
    """
    pop = []
    
    # 1. Standard 4x4 Grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    pop.append([p[:] for p in base_grid])
    
    # 2. Sheared Grids
    for _ in range(5):
        shear = rng.randint(1, 3)
        pts = []
        for x in range(4):
            for y in range(4):
                nx = x + shear * y
                ny = y
                # Scale to fit bounds
                pts.append([nx, ny])
        # Randomize slightly
        for i in range(16):
            pts[i][0] += rng.randint(-2, 2)
            pts[i][1] += rng.randint(-2, 2)
        pop.append(pts)
        
    # 3. Random Points
    for _ in range(10):
        pts = []
        while len(pts) < 16:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        pop.append(pts)
        
    # 4. "Diagonals" focus
    pts = []
    # 4 points on y=x, 4 on y=-x, 4 on y=x+2, 4 on y=x-2
    # Need to ensure distinctness
    # Just add random valid points
    for _ in range(16):
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if [x, y] not in pts:
            pts.append([x, y])
    pop.append(pts)

    return pop

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    # Initial Population
    population = _initialize_population(rng)
    
    # Evaluate initial population
    scored_pop = []
    for ind in population:
        score, has_5, _ = _score_lines(ind)
        if has_5:
            # Try to repair
            repaired = _repair(ind, rng)
            if repaired:
                score, has_5, _ = _score_lines(repaired)
                if not has_5:
                    ind = repaired
                else:
                    score = -1000 # Penalize invalid
            else:
                score = -1000
        scored_pop.append((score, ind))
        
    # Sort by score descending
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_pop[0][0]
    best_points = scored_pop[0][1]
    
    # Genetic Algorithm Loop
    pop_size = 20
    generations = 0
    
    while time.time() < deadline - 10:
        generations += 1
        
        # Selection: Tournament of 3
        def select():
            candidates = rng.sample(scored_pop, 3)
            return max(candidates, key=lambda x: x[0])[1]
            
        new_population = []
        
        # Elitism: Keep top 2
        for i in range(2):
            new_population.append((scored_pop[i][0], scored_pop[i][1]))
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            p1 = select()
            p2 = select()
            
            child = _crossover(p1, p2, rng)
            
            # Mutation
            r = rng.random()
            if r < 0.3:
                child = _mutate_complete(child, rng)
            elif r < 0.6:
                child = _mutate_perturb(child, rng)
            else:
                # No mutation
                pass
                
            # Repair if invalid
            score, has_5, _ = _score_lines(child)
            if has_5:
                repaired = _repair(child, rng)
                if repaired:
                    child = repaired
                    score, has_5, _ = _score_lines(child)
                    if has_5:
                        score = -1000
                else:
                    score = -1000
            
            new_population.append((score, child))
            
        # Update population
        scored_pop = new_population
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_points = scored_pop[0][1]
            
        # Optional: Local search on the best individual periodically
        if generations % 10 == 0:
            # Run a quick local search on the current best
            # This is a hybrid GA + LS
            current_best = scored_pop[0][1]
            # Simple LS: try all single-point moves for the best individual?
            # Too slow. Just stick to GA for now.
            
    # Final check
    final_score, final_has_5, _ = _score_lines(best_points)
    if final_has_5:
        # Fallback to a known valid grid if something went wrong
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
# EVOLVE-BLOCK-END
