"""Optimized program using finite projective plane construction and local search
to minimize the largest 3-free subset while respecting the 'no four collinear' constraint."""
import random
import math
import time
from itertools import combinations

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # seconds (evaluator kills at 150s, keep safety margin)

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    """Check if three points are collinear."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear.
    points: list of tuples (x, y). Returns the size of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity for each pair
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j:
                    if _is_collinear(points[i], points[j], points[k]):
                        collinearity[i][j] |= (1 << k)
                        collinearity[j][i] |= (1 << k)
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound: size + (n - i)
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinearity[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j:
                    if _is_collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return True
    return False

def _generate_projective_plane_points():
    """Generate 20 points from a finite projective plane-like structure.
    
    We use a construction based on placing points to maximize collinear triples
    while avoiding 4-collinearities.
    """
    random.seed(12345)
    
    # Strategy: Create a configuration with many 3-point lines
    # Use a grid-based approach with strategic perturbations
    
    # Base points on a 5x4 grid with offsets to create collinear triples
    points = []
    
    # Create rows with 3-4 points each, with offsets to create diagonal collinearities
    for row in range(4):
        y = row * 20
        for col in range(5):
            x = col * 20 + row * 3  # offset to avoid perfect grid
            points.append((x, y))
    
    # We now have 20 points. Check for 4-collinear and fix if needed
    while _has_four_collinear(points):
        # Move a random point slightly
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        for _ in range(100):
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                if (new_x, new_y) not in points:
                    points[idx] = (new_x, new_y)
                    if not _has_four_collinear(points):
                        break
        else:
            # Reset if we couldn't fix
            points[idx] = (old_x, old_y)
    
    return points

def _generate_parabola_points():
    """Generate points on a parabola with perturbations."""
    random.seed(99999)
    points = []
    
    # Points on y = x^2 / 4 with perturbations
    for i in range(-10, 10):
        x = i
        y = (x * x) // 4 + random.randint(-2, 2)
        points.append((x, y))
    
    # Ensure no duplicates and no 4-collinear
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)
    points = unique_points[:20]
    
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in points:
            points.append((x, y))
    
    while _has_four_collinear(points):
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        for _ in range(100):
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                if (new_x, new_y) not in points:
                    points[idx] = (new_x, new_y)
                    if not _has_four_collinear(points):
                        break
        else:
            points[idx] = (old_x, old_y)
    
    return points

def _generate_random_valid():
    """Generate a random valid configuration."""
    random.seed(77777)
    points = []
    for _ in range(20):
        for _ in range(1000):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in points:
                points.append((x, y))
                break
    
    while _has_four_collinear(points):
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        for _ in range(200):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if (new_x, new_y) not in points:
                points[idx] = (new_x, new_y)
                if not _has_four_collinear(points):
                    break
        else:
            points[idx] = (old_x, old_y)
    
    return points

def _local_search(points, max_time=100.0):
    """Local search to minimize the 3-free subset size."""
    random.seed(54321)
    
    current_points = list(points)
    current_score = _max_3_free_subset(current_points)
    best_score = current_score
    best_points = list(current_points)
    
    start_time = time.time()
    temperature = 5.0
    cooling_rate = 0.995
    min_temperature = 0.1
    
    while _time_left() > 10.0 and temperature > min_temperature:
        # Try to move a random point
        idx = random.randint(0, 19)
        old_point = current_points[idx]
        
        # Try several random new positions
        accepted = False
        for _ in range(15):
            # Move point to a random nearby position
            move_range = random.choice([3, 5, 10])
            new_point = (old_point[0] + random.randint(-move_range, move_range),
                        old_point[1] + random.randint(-move_range, move_range))
            
            if new_point[0] < -100 or new_point[0] > 100 or new_point[1] < -100 or new_point[1] > 100:
                continue
            if new_point in current_points:
                continue
            
            current_points[idx] = new_point
            
            if _has_four_collinear(current_points):
                current_points[idx] = old_point
                continue
            
            new_score = _max_3_free_subset(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(current_points)
                accepted = True
                break
            elif new_score == current_score:
                if random.random() < 0.2:
                    accepted = True
                    break
            else:
                delta = new_score - current_score
                if random.random() < math.exp(-delta / temperature):
                    current_score = new_score
                    accepted = True
                    break
        
        if not accepted:
            current_points[idx] = old_point
        
        temperature *= cooling_rate
    
    return best_points, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Try multiple initial configurations and pick the best
    generators = [
        _generate_projective_plane_points,
        _generate_parabola_points,
        _generate_random_valid,
    ]
    
    best_score = float('inf')
    best_points = None
    
    # Try each generator with local search
    for gen_func in generators:
        if _time_left() < 30.0:
            break
        
        try:
            init_points = gen_func()
            if best_points is None or _max_3_free_subset(init_points) < best_score:
                best_score = _max_3_free_subset(init_points)
                best_points = list(init_points)
            
            # Run local search
            search_points, search_score = _local_search(init_points, max_time=30.0)
            if search_score < best_score:
                best_score = search_score
                best_points = list(search_points)
        except Exception:
            continue
    
    # Final local search on the best found
    if best_points is not None and _time_left() > 20.0:
        final_points, final_score = _local_search(best_points, max_time=_time_left() - 15.0)
        if final_score < best_score:
            best_score = final_score
            best_points = list(final_points)
    
    if best_points is None:
        # Fallback: simple random
        best_points = _generate_random_valid()
    
    return [list(p) for p in best_points]
