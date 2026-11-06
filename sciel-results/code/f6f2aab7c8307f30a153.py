import numpy as np
import time
import random
from collections import Counter

# Pre-compute triple indices for 30 points
def _get_triple_indices(n=30):
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

_TRIPLE_I, _TRIPLE_J, _TRIPLE_K = _get_triple_indices(30)
_N_TRIPLES = len(_TRIPLE_I)

def _vectorized_compute_areas(points):
    """
    Compute absolute doubled areas for all triples of 30 points.
    points: numpy array of shape (30, 2)
    Returns: numpy array of shape (4500,) with absolute determinants.
    """
    pi = points[_TRIPLE_I]
    pj = points[_TRIPLE_J]
    pk = points[_TRIPLE_K]
    dx_ji = pj[:, 0] - pi[:, 0]
    dy_ji = pj[:, 1] - pi[:, 1]
    dx_ki = pk[:, 0] - pi[:, 0]
    dy_ki = pk[:, 1] - pi[:, 1]
    dets = dx_ji * dy_ki - dy_ji * dx_ki
    return np.abs(dets)

def _get_max_multiplicity(areas):
    """
    Given an array of areas, return (max_count, most_frequent_area).
    Ignores zero areas.
    """
    if len(areas) == 0:
        return 0, 0
    non_zero = areas[areas > 0]
    if len(non_zero) == 0:
        return 0, 0
    unique_areas, counts = np.unique(non_zero, return_counts=True)
    max_idx = np.argmax(counts)
    return counts[max_idx], unique_areas[max_idx]

def _count_area(areas, target):
    """Count how many times target area appears in areas."""
    return np.sum(areas == target)

def _random_points_in_box(n=30, box=30):
    """Generate n distinct random points in [-box, box]^2."""
    points = set()
    while len(points) < n:
        x = random.randint(-box, box)
        y = random.randint(-box, box)
        points.add((x, y))
    return np.array(list(points), dtype=np.int32)

def _grid_points(rows, cols, offset_x=0, offset_y=0):
    """Generate a grid of points."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = i * 2 + offset_x
            y = j * 2 + offset_y
            pts.append((x, y))
    return np.array(pts[:30], dtype=np.int32)

def _structured_points():
    """Generate structured points that might have good area properties."""
    patterns = []
    patterns.append(_grid_points(6, 5, -5, -4))
    patterns.append(_grid_points(5, 6, -4, -5))
    patterns.append(_random_points_in_box())
    pts = []
    for i in range(7):
        for j in range(5):
            if len(pts) < 30:
                pts.append((i * 3 - 9, j * 3 - 6))
    patterns.append(np.array(pts, dtype=np.int32))
    return patterns

def _incremental_move(areas, points, idx, new_point):
    """
    Update the areas array after moving point idx to new_point.
    Returns the updated areas array (modified in place for speed).
    """
    old_x, old_y = int(points[idx, 0]), int(points[idx, 1])
    new_x, new_y = int(new_point[0]), int(new_point[1])
    
    # Find all triples involving idx
    mask_i = _TRIPLE_I == idx
    mask_j = _TRIPLE_J == idx
    mask_k = _TRIPLE_K == idx
    mask = mask_i | mask_j | mask_k
    
    # Recompute only those triples
    if np.any(mask):
        ti = _TRIPLE_I[mask]
        tj = _TRIPLE_J[mask]
        tk = _TRIPLE_K[mask]
        
        pi = points[ti]
        pj = points[tj]
        pk = points[tk]
        
        dx_ji = pj[:, 0] - pi[:, 0]
        dy_ji = pj[:, 1] - pi[:, 1]
        dx_ki = pk[:, 0] - pi[:, 0]
        dy_ki = pk[:, 1] - pi[:, 1]
        
        new_dets = dx_ji * dy_ki - dy_ji * dx_ki
        areas[mask] = np.abs(new_dets)
    
    return areas

def _sa_search(points, target_area, deadline, initial_temp=3.0, dynamic_target=True):
    """
    Simulated annealing with incremental updates and dynamic target switching.
    Returns (best_score, best_points, best_area).
    """
    curr_points = points.copy()
    curr_occupied = set(map(tuple, curr_points.tolist()))
    
    curr_areas = _vectorized_compute_areas(curr_points)
    
    # Track multiplicity of current target
    curr_target_count = _count_area(curr_areas, target_area)
    curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
    
    # If dynamic target, use the current max area as target
    if dynamic_target:
        target_area = int(curr_max_area)
        curr_target_count = _count_area(curr_areas, target_area)
    
    best_score = max(curr_target_count, curr_max_mult)
    best_points = curr_points.copy()
    best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
    
    temp = initial_temp
    cooling = 0.99995
    min_temp = 0.003
    
    no_improve_count = 0
    max_no_improve = 8000
    steps = 0
    last_recompute = 0
    recompute_interval = 2000
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
        
        # Generate new position
        r = random.random()
        if r < 0.35:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.75:
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        else:
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        
        new_point = (new_x, new_y)
        if new_point == (old_x, old_y):
            continue
        if new_point in curr_occupied:
            continue
            
        # Perform move
        curr_points[idx] = [new_x, new_y]
        curr_occupied.remove((old_x, old_y))
        curr_occupied.add(new_point)
        
        # Incremental update
        _incremental_move(curr_areas, curr_points, idx, new_point)
        
        # Recompute full areas periodically to avoid drift
        if steps - last_recompute >= recompute_interval:
            curr_areas = _vectorized_compute_areas(curr_points)
            last_recompute = steps
        
        # Evaluate
        new_target_count = _count_area(curr_areas, target_area)
        new_max_mult, new_max_area = _get_max_multiplicity(curr_areas)
        
        # Dynamic target switching: if the max area changed, switch target
        if dynamic_target and new_max_area != target_area:
            target_area = int(new_max_area)
            new_target_count = _count_area(curr_areas, target_area)
        
        # Score: primarily target count, but also consider max multiplicity
        new_score = max(new_target_count, new_max_mult)
        curr_score = max(curr_target_count, curr_max_mult)
        
        improved = False
        if new_score > curr_score:
            improved = True
            no_improve_count = 0
        elif new_score == curr_score:
            if new_target_count > curr_target_count:
                improved = True
                no_improve_count = 0
            elif random.random() < 0.5:
                improved = True
        else:
            delta = curr_score - new_score
            if random.random() < np.exp(-delta / temp):
                improved = True
        
        if improved:
            curr_target_count = new_target_count
            curr_max_mult = new_max_mult
            curr_max_area = new_max_area
            curr_score = new_score
            
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points.copy()
                best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            curr_points[idx] = [old_x, old_y]
            curr_occupied.remove(new_point)
            curr_occupied.add((old_x, old_y))
            _incremental_move(curr_areas, curr_points, idx, (old_x, old_y))
            no_improve_count += 1
        
        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        
        # If stuck for too long, restart with higher temp
        if no_improve_count > max_no_improve:
            temp = max(temp, 1.5)
            no_improve_count = 0
    
    return best_score, best_points, best_area

def _polish(points, deadline, initial_temp=1.0):
    """
    Final high-precision polish pass with tighter step distribution.
    """
    curr_points = points.copy()
    curr_occupied = set(map(tuple, curr_points.tolist()))
    curr_areas = _vectorized_compute_areas(curr_points)
    curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
    target_area = int(curr_max_area)
    curr_target_count = _count_area(curr_areas, target_area)
    
    best_score = max(curr_target_count, curr_max_mult)
    best_points = curr_points.copy()
    best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
    
    temp = initial_temp
    cooling = 0.99999
    min_temp = 0.001
    no_improve_count = 0
    max_no_improve = 10000
    steps = 0
    last_recompute = 0
    recompute_interval = 3000
    
    while time.time() < deadline:
        steps += 1
        idx = random.randint(0, 29)
        old_x, old_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
        
        r = random.random()
        if r < 0.25:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.80:
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        else:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        
        new_point = (new_x, new_y)
        if new_point == (old_x, old_y) or new_point in curr_occupied:
            continue
        
        curr_points[idx] = [new_x, new_y]
        curr_occupied.remove((old_x, old_y))
        curr_occupied.add(new_point)
        _incremental_move(curr_areas, curr_points, idx, new_point)
        
        if steps - last_recompute >= recompute_interval:
            curr_areas = _vectorized_compute_areas(curr_points)
            last_recompute = steps
        
        new_target_count = _count_area(curr_areas, target_area)
        new_max_mult, new_max_area = _get_max_multiplicity(curr_areas)
        
        if new_max_area != target_area:
            target_area = int(new_max_area)
            new_target_count = _count_area(curr_areas, target_area)
        
        new_score = max(new_target_count, new_max_mult)
        curr_score = max(curr_target_count, curr_max_mult)
        
        improved = False
        if new_score > curr_score:
            improved = True
            no_improve_count = 0
        elif new_score == curr_score:
            if new_target_count > curr_target_count:
                improved = True
                no_improve_count = 0
            elif random.random() < 0.5:
                improved = True
        else:
            delta = curr_score - new_score
            if random.random() < np.exp(-delta / temp):
                improved = True
        
        if improved:
            curr_target_count = new_target_count
            curr_max_mult = new_max_mult
            curr_max_area = new_max_area
            curr_score = new_score
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points.copy()
                best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            curr_points[idx] = [old_x, old_y]
            curr_occupied.remove(new_point)
            curr_occupied.add((old_x, old_y))
            _incremental_move(curr_areas, curr_points, idx, (old_x, old_y))
            no_improve_count += 1
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        if no_improve_count > max_no_improve:
            temp = max(temp, 1.0)
            no_improve_count = 0
    
    return best_score, best_points, best_area

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Hard cap: 55s to stay well under 60s limit
    
    # Try multiple initial configurations
    initial_configs = _structured_points()
    
    best_overall_score = 0
    best_overall_points = None
    best_overall_area = 0
    
    # Phase 1: Quick evaluation of initial configs
    for config in initial_configs:
        if time.time() > deadline:
            break
        areas = _vectorized_compute_areas(config)
        score, area = _get_max_multiplicity(areas)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = config.copy()
            best_overall_area = area
    
    if best_overall_points is None:
        best_overall_points = _random_points_in_box()
        areas = _vectorized_compute_areas(best_overall_points)
        best_overall_score, best_overall_area = _get_max_multiplicity(areas)

    # Phase 2: Main SA search with dynamic target (30s budget)
    sub_deadline_2 = min(deadline - 25, start_time + 30)
    if time.time() < sub_deadline_2:
        score, points, area = _sa_search(best_overall_points, best_overall_area, sub_deadline_2, initial_temp=3.0, dynamic_target=True)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_area = area
    
    # Phase 3: Final polish pass (remaining time, at least 15s)
    if time.time() < deadline - 2:
        polish_deadline = deadline - 1
        score, points, area = _polish(best_overall_points, polish_deadline, initial_temp=0.8)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_area = area

    if best_overall_points is None:
        best_overall_points = _random_points_in_box()

    return [(int(x), int(y)) for x, y in best_overall_points.tolist()]
