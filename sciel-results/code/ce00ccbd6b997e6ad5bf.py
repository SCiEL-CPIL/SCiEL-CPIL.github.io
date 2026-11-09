import time
import random
import itertools
import sys

sys.setrecursionlimit(10000)

def _are_collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_collinear(points[i], points[j], points[k]) and \
                       _are_collinear(points[i], points[j], points[l]):
                        return True
    return False

def _max_no_three_collinear(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound.
    """
    n = len(points)
    
    # Precompute collinearity: collinear[i][j][k] = True if points i,j,k are collinear
    collinear = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                collinear[(i, j, k)] = _are_collinear(points[i], points[j], points[k])
    
    # Greedy initial solution: try to build a large no-three-collinear set
    # Start with all points, remove violations
    # Actually, let's use a simpler approach: branch and bound
    
    best = [0]
    
    # Sort points by index for consistent ordering
    # Branch and bound: try to include or exclude each point
    # State: current set (as bitmask or list), index we're considering
    
    # For n=20, we need an efficient approach
    # Use the fact that we're looking for max subset with no 3 collinear
    
    # Greedy heuristic for upper bound: 
    # Process points, add if it doesn't create a triple with existing
    def greedy_upper_bound(remaining_indices, current_set):
        """Upper bound: assume we can add all remaining that don't immediately conflict."""
        ub = len(current_set)
        for idx in remaining_indices:
            # Check if adding idx creates a triple with any two in current_set
            conflict = False
            for a, b in itertools.combinations(current_set, 2):
                if collinear.get(tuple(sorted((idx, a, b))), False):
                    conflict = True
                    break
            if not conflict:
                ub += 1
        return ub
    
    # Branch and bound
    def branch_and_bound(idx, current_set, excluded):
        if idx >= n:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
            return
        
        # Prune: even if we add all remaining, can't beat best
        remaining = n - idx
        if len(current_set) + remaining <= best[0]:
            return
        
        # Option 1: exclude point idx
        excluded.add(idx)
        branch_and_bound(idx + 1, current_set, excluded)
        excluded.remove(idx)
        
        # Option 2: include point idx if it doesn't create a triple
        can_include = True
        for a, b in itertools.combinations(current_set, 2):
            if collinear.get(tuple(sorted((idx, a, b))), False):
                can_include = False
                break
        
        if can_include:
            current_set.add(idx)
            branch_and_bound(idx + 1, current_set, excluded)
            current_set.remove(idx)
    
    # Initial greedy to get a good lower bound for best
    # Try multiple orderings
    current_set = set()
    for i in range(n):
        can_add = True
        for a, b in itertools.combinations(current_set, 2):
            if collinear.get(tuple(sorted((i, a, b))), False):
                can_add = False
                break
        if can_add:
            current_set.add(i)
    best[0] = len(current_set)
    
    # Now branch and bound to find the true maximum
    # But this might be too slow for n=20 in worst case
    # Let's use a smarter approach with memoization or better pruning
    
    # Actually for n=20, branch and bound with good pruning should work
    # Let's add a time limit
    start_time = time.time()
    timeout = 5.0
    
    current_set = set()
    excluded = set()
    branch_and_bound(0, current_set, excluded)
    
    return best[0]

def _compute_score(points):
    """Compute the score: largest subset with no three collinear."""
    return _max_no_three_collinear(points)

def _is_valid(points):
    """Check if configuration is valid: 20 distinct points, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    if len(set(map(tuple, points))) != 20:
        return False
    for x, y in points:
        if abs(x) > 100 or abs(y) > 100:
            return False
    if _has_four_collinear(points):
        return False
    return True

def _random_valid_point(occupied):
    """Generate a random point not in occupied set, within bounds."""
    while True:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in occupied:
            return [x, y]

def _try_move_point(points, idx, deadline):
    """
    Try to move point idx to a new position to improve (reduce) the score.
    Returns the new points if improved, else None.
    """
    original = points[idx][:]
    current_score = _compute_score(points)
    
    # Try random new positions
    occupied = set(map(tuple, points))
    occupied.discard(tuple(original))
    
    best_points = None
    best_score = current_score
    
    for _ in range(50):
        if time.time() > deadline:
            break
        new_pt = _random_valid_point(occupied)
        new_points = points[:]
        new_points[idx] = new_pt
        
        if not _is_valid(new_points):
            continue
        
        new_score = _compute_score(new_points)
        if new_score < best_score:
            best_score = new_score
            best_points = new_points
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s budget, leave margin
    
    random.seed(42)
    
    # Initial random configuration
    points = []
    occupied = set()
    for _ in range(20):
        while True:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in occupied:
                occupied.add((x, y))
                points.append([x, y])
                break
    
    # Ensure no four collinear
    attempts = 0
    while _has_four_collinear(points) and attempts < 1000:
        # Move a random point
        idx = random.randint(0, 19)
        old = points[idx][:]
        occupied.discard(tuple(old))
        new_pt = _random_valid_point(occupied)
        points[idx] = new_pt
        occupied.add(tuple(new_pt))
        attempts += 1
    
    if _has_four_collinear(points):
        # Fallback: use a known valid configuration
        # Parabola points (no three collinear, so no four)
        return [[x, x * x] for x in range(-10, 10)]
    
    best_points = points[:]
    best_score = _compute_score(best_points)
    
    # Local search
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        
        # Try moving each point
        improved = False
        for idx in range(20):
            if time.time() > deadline - 10:
                break
            
            new_points = _try_move_point(best_points, idx, deadline - 10)
            if new_points is not None:
                new_score = _compute_score(new_points)
                if new_score < best_score:
                    best_points = new_points
                    best_score = new_score
                    improved = True
        
        # If no improvement, try a random restart or perturbation
        if not improved:
            # Perturbation: move 3 random points
            if time.time() > deadline - 20:
                break
            for _ in range(3):
                idx = random.randint(0, 19)
                old = best_points[idx][:]
                occupied = set(map(tuple, best_points))
                occupied.discard(tuple(old))
                new_pt = _random_valid_point(occupied)
                best_points[idx] = new_pt
                occupied.add(tuple(new_pt))
            
            # Check validity
            if not _is_valid(best_points):
                # Revert to previous best
                continue
            
            new_score = _compute_score(best_points)
            if new_score < best_score:
                best_score = new_score
    
    return best_points
