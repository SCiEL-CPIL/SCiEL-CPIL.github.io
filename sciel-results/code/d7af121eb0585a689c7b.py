"""Improved program: Hypergraph-based optimization with incremental collinearity tracking
and direct minimization of the 3-free subset size using structured initial configurations."""
import random
import math
import time
from collections import defaultdict

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # seconds (safety margin before 150s kill)

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _has_four_collinear(points):
    """Check if any four points are collinear. Returns the offending quadruple or None."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples. O(n^3) but with early exit for 4-collinear check."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute for each pair (i,j), the set of points collinear with both
    collinear_with_pair = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_with_pair[i][j] = mask
            collinear_with_pair[j][i] = mask
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i - check if it creates a collinear triple
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinear_with_pair[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _generate_parabola_points():
    """Generate points on a parabola y = x^2, which has no 3 collinear.
    Then perturb to create collinearities while keeping no 4 collinear."""
    points = []
    # Use x values that give small coordinates
    x_vals = list(range(-4, 5))  # -4 to 4, gives 9 points
    # Add more with different scaling
    for x in x_vals:
        points.append((x, x * x))
    # Need 20 points, add more
    for x in range(5, 10):
        points.append((x, x * x))
    # That's 14, need 6 more
    for x in range(-10, -4):
        points.append((x, x * x))
    # That's 20
    return points[:20]

def _generate_grid_perturbed():
    """Generate a perturbed grid that has many near-collinearities."""
    random.seed(42)
    points = []
    # Start with a 5x4 grid
    base_grid = []
    for x in range(5):
        for y in range(4):
            base_grid.append((x * 8 - 16, y * 8 - 12))
    
    # Perturb each point slightly
    for i, (x, y) in enumerate(base_grid):
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        points.append((x + dx, y + dy))
    
    return points

def _fix_four_collinear(points):
    """Try to fix any 4-collinear violations by moving points."""
    attempts = 0
    while _has_four_collinear(points) and attempts < 5000:
        # Find a random point and move it
        idx = random.randint(0, 19)
        old = points[idx]
        new = (random.randint(-30, 30), random.randint(-30, 30))
        if new not in points and abs(new[0]) <= 100 and abs(new[1]) <= 100:
            points[idx] = new
        attempts += 1
    return points

def _delta_triples(points, idx, old_point, new_point):
    """Compute the change in number of collinear triples when moving point idx from old to new."""
    n = len(points)
    old_count = 0
    new_count = 0
    
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            # Old: check if old_point, points[i], points[j] are collinear
            if _is_collinear(old_point, points[i], points[j]):
                old_count += 1
            # New: check if new_point, points[i], points[j] are collinear
            if _is_collinear(new_point, points[i], points[j]):
                new_count += 1
    
    return new_count - old_count

def _check_new_point_valid(points, idx, new_point):
    """Check if placing new_point at idx keeps no 4 collinear."""
    # Temporarily place new_point
    old = points[idx]
    points[idx] = new_point
    valid = not _has_four_collinear(points)
    points[idx] = old
    return valid

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Parabola-based
    initial_configs.append(_generate_parabola_points())
    
    # Config 2: Perturbed grid
    initial_configs.append(_generate_grid_perturbed())
    
    # Config 3: Random in small box
    pts3 = []
    for _ in range(20):
        p = (random.randint(-15, 15), random.randint(-15, 15))
        while p in pts3:
            p = (random.randint(-15, 15), random.randint(-15, 15))
        pts3.append(p)
    initial_configs.append(pts3)
    
    # Config 4: Structured - use points that create many 3-collinear lines
    # e.g., 4 lines with 5 points each, but no 4 collinear
    # Actually, let's use a known construction: take points from a 2D lattice
    # but break 4-collinearities
    pts4 = []
    # Use a "staircase" pattern
    for i in range(20):
        x = i % 7
        y = i // 7
        pts4.append((x * 5 - 10, y * 5 - 5))
    initial_configs.append(pts4)
    
    # Fix all configs and evaluate
    best_points = None
    best_score = 21  # impossible, will be replaced
    
    for config in initial_configs:
        if _time_left() < 5.0:
            break
        config = _fix_four_collinear(config)
        if _has_four_collinear(config):
            continue
        
        score = _max_3_free_subset(config)
        if score < best_score:
            best_score = score
            best_points = list(config)
    
    if best_points is None:
        # Fallback: random
        best_points = []
        for _ in range(20):
            p = (random.randint(-100, 100), random.randint(-100, 100))
            while p in best_points:
                p = (random.randint(-100, 100), random.randint(-100, 100))
            best_points.append(p)
        while _has_four_collinear(best_points):
            idx = random.randint(0, 19)
            best_points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
            while best_points[idx] in best_points:
                best_points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
        best_score = _max_3_free_subset(best_points)
    
    # Local search: directly minimize the 3-free subset size
    points = list(best_points)
    current_score = best_score
    current_triples = _count_collinear_triples(points)
    
    # Simulated annealing on the exact score with incremental triple counting
    temperature = 5.0
    cooling_rate = 0.998
    min_temperature = 0.01
    no_improve_count = 0
    max_no_improve = 2000
    
    while temperature > min_temperature and _time_left() > 15.0 and no_improve_count < max_no_improve:
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Generate new point: biased toward existing points to create collinearities
        if random.random() < 0.5:
            # Near another point
            other_idx = random.randint(0, 19)
            if other_idx == idx:
                other_idx = (other_idx + 1) % 20
            other = points[other_idx]
            new_point = (other[0] + random.randint(-3, 3), other[1] + random.randint(-3, 3))
        else:
            # Random in box
            new_point = (random.randint(-25, 25), random.randint(-25, 25))
        
        if new_point in points:
            continue
        if abs(new_point[0]) > 100 or abs(new_point[1]) > 100:
            continue
        
        # Check validity
        points[idx] = new_point
        if _has_four_collinear(points):
            points[idx] = old_point
            continue
        
        # Compute delta in triples
        delta_triples = _delta_triples(points, idx, old_point, new_point)
        
        # Compute new exact score (expensive, but we need it for the objective)
        # Optimization: only compute if delta suggests improvement
        if delta_triples > 0 or random.random() < 0.1:  # Compute score more often if triples increase
            new_score = _max_3_free_subset(points)
            delta_score = new_score - current_score
            
            if delta_score <= 0:
                # Improvement or equal
                current_score = new_score
                current_triples += delta_triples
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Worse, accept with probability
                if random.random() < math.exp(-delta_score / temperature):
                    current_score = new_score
                    current_triples += delta_triples
                    no_improve_count += 1
                else:
                    points[idx] = old_point
                    no_improve_count += 1
        else:
            # Don't compute exact score, just track triples as proxy
            if delta_triples > 0:
                current_triples += delta_triples
            else:
                points[idx] = old_point
            no_improve_count += 1
        
        temperature *= cooling_rate
    
    # Final refinement: try targeted moves
    points = list(best_points)
    current_score = best_score
    
    for iteration in range(500):
        if _time_left() < 5.0:
            break
        
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try to place point to create a new collinear triple
        # Find two existing points and try to place near the line
        i = random.randint(0, 19)
        j = random.randint(0, 19)
        if i == idx or j == idx or i == j:
            continue
        
        p_i = points[i]
        p_j = points[j]
        
        # Try points on the line through p_i and p_j
        dx = p_j[0] - p_i[0]
        dy = p_j[1] - p_i[1]
        
        # Try a few positions on or near this line
        found = False
        for t in range(-2, 3):
            if t == 0:
                continue
            new_x = p_i[0] + t * dx
            new_y = p_i[1] + t * dy
            if abs(new_x) <= 100 and abs(new_y) <= 100 and (new_x, new_y) not in points:
                points[idx] = (new_x, new_y)
                if not _has_four_collinear(points):
                    new_score = _max_3_free_subset(points)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = list(points)
                    else:
                        points[idx] = old_point
                    found = True
                    break
        
        if not found:
            # Try random small perturbation
            new_point = (old_point[0] + random.randint(-2, 2), old_point[1] + random.randint(-2, 2))
            if new_point not in points and abs(new_point[0]) <= 100 and abs(new_point[1]) <= 100:
                points[idx] = new_point
                if not _has_four_collinear(points):
                    new_score = _max_3_free_subset(points)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = list(points)
                    else:
                        points[idx] = old_point
    
    return [list(p) for p in best_points]
