# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import product
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _line_from_points(p1, p2):
    """Get normalized line from two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x2
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """
    Count incidences between points and lines.
    points: (N, 2) array
    lines: (M, 3) array
    Returns total incidences.
    """
    # Vectorized check: A*x + B*y + C == 0
    # points: (N, 2) -> (N, 1) for broadcasting? 
    # lines: (M, 3)
    # We want (N, M) boolean matrix.
    # p_x = points[:, 0:1] (N, 1)
    # p_y = points[:, 1:2] (N, 1)
    # l_A = lines[:, 0:1] (M, 1) -> transpose to (1, M)
    # l_B = lines[:, 1:2] (M, 1) -> transpose to (1, M)
    # l_C = lines[:, 2:3] (M, 1) -> transpose to (1, M)
    
    px = points[:, 0:1] # (N, 1)
    py = points[:, 1:2] # (N, 1)
    lA = lines[:, 0:1].T # (1, M)
    lB = lines[:, 1:2].T # (1, M)
    lC = lines[:, 2:3].T # (1, M)
    
    # (N, 1) * (1, M) -> (N, M)
    vals = px * lA + py * lB + lC
    incs = np.sum(vals == 0)
    return incs

def _generate_line_pool(max_coord=10):
    """
    Generate a pool of candidate lines with small integer coefficients.
    These are lines likely to form dense configurations.
    """
    lines = set()
    # Horizontal, Vertical, Diagonals
    for c in range(-max_coord, max_coord + 1):
        lines.add(_normalize_line(1, 0, -c)) # x = c
        lines.add(_normalize_line(0, 1, -c)) # y = c
        lines.add(_normalize_line(1, 1, -c)) # x + y = c
        lines.add(_normalize_line(1, -1, -c)) # x - y = c
        
    # Other simple slopes: 2, -2, 1/2, -1/2
    for c in range(-max_coord * 2, max_coord * 2 + 1):
        lines.add(_normalize_line(2, 1, -c)) # 2x + y = c
        lines.add(_normalize_line(1, 2, -c)) # x + 2y = c
        lines.add(_normalize_line(2, -1, -c)) # 2x - y = c
        lines.add(_normalize_line(1, -2, -c)) # x - 2y = c
        
    return list(lines)

def _find_best_lines_and_points(pool_lines, num_lines=27, num_points=54, time_limit=580):
    """
    1. Select num_lines from pool_lines to maximize intersection potential.
    2. Select num_points from intersections to maximize incidences.
    """
    start_time = time.time()
    
    # Step 1: Greedy selection of lines
    # We want lines that intersect each other at integer points.
    # Two lines (A1, B1, C1) and (A2, B2, C2) intersect at integer point if:
    # det = A1*B2 - A2*B1
    # x = (B1*C2 - B2*C1) / det
    # y = (A2*C1 - A1*C2) / det
    # x, y must be integers.
    
    # Instead of complex intersection checks, let's just pick a diverse set of lines
    # and then compute the best points for them.
    
    # Strategy: 
    # 1. Pick a base set of lines (e.g., grid lines).
    # 2. Compute all integer intersection points of these lines.
    # 3. For each intersection point, count how many of the selected lines pass through it.
    # 4. Select the top 54 points with highest degree.
    # 5. The score is the sum of degrees of selected points.
    
    # We can improve the line selection by swapping lines to increase the number
    # of high-degree intersection points.
    
    # Initial set: First 27 lines from pool (mostly grid-like)
    # Let's sort pool by "simplicity" or just take a mix.
    random.shuffle(pool_lines)
    current_lines = pool_lines[:num_lines]
    
    # Function to evaluate a set of lines
    def evaluate_lines(lines_set):
        # Find all integer intersection points
        # N_lines = 27 -> 351 pairs
        # For each pair, check if intersection is integer.
        # If so, add point to dict: point -> count of lines passing through it?
        # No, count of lines in the set that pass through the point.
        
        # Efficient way:
        # 1. Collect all candidate points (intersections of pairs in lines_set).
        # 2. For each candidate point, count how many lines in lines_set pass through it.
        # 3. Sort points by count, take top 54.
        # 4. Sum counts.
        
        candidates = set()
        ls = list(lines_set)
        n_l = len(ls)
        
        for i in range(n_l):
            A1, B1, C1 = ls[i]
            for j in range(i + 1, n_l):
                A2, B2, C2 = ls[j]
                det = A1 * B2 - A2 * B1
                if det == 0:
                    continue # Parallel
                # x = (B1*C2 - B2*C1) / det
                # y = (A2*C1 - A1*C2) / det
                num_x = B1 * C2 - B2 * C1
                num_y = A2 * C1 - A1 * C2
                
                if num_x % det == 0 and num_y % det == 0:
                    x = num_x // det
                    y = num_y // det
                    # Limit coordinate range to avoid huge numbers? 
                    # Small coefficients -> small intersections usually.
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.add((x, y))
        
        if not candidates:
            return 0, []
            
        # Count incidences for each candidate point
        point_degrees = {}
        for p in candidates:
            x, y = p
            deg = 0
            for l in ls:
                A, B, C = l
                if A * x + B * y + C == 0:
                    deg += 1
            if deg > 0:
                point_degrees[p] = deg
                
        # Select top 54 points
        sorted_points = sorted(point_degrees.items(), key=lambda item: item[1], reverse=True)
        selected_points = [p for p, d in sorted_points[:num_points]]
        score = sum(d for p, d in sorted_points[:num_points])
        
        return score, selected_points

    current_score, current_points = evaluate_lines(current_lines)
    best_score = current_score
    best_lines = current_lines
    best_points = current_points
    
    # Local search on lines: Swap a line in current_lines with a line from the rest of the pool
    # to see if it improves the score.
    
    pool_set = set(pool_lines)
    current_set = set(current_lines)
    remaining_lines = list(pool_set - current_set)
    
    # To speed up, we can pre-filter remaining lines to those that intersect
    # many current lines at integer points.
    
    steps = 0
    while time.time() - start_time < time_limit:
        # Pick a random line from current_set to remove
        l_out = random.choice(list(current_set))
        # Pick a random line from remaining_lines to add
        l_in = random.choice(remaining_lines)
        
        new_set = (current_set - {l_out}) | {l_in}
        
        # Evaluate
        new_score, new_points = evaluate_lines(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_lines = list(new_set)
            best_points = new_points
            current_set = new_set
            # Update remaining
            remaining_lines = list(pool_set - current_set)
            
        elif new_score > current_score:
            # Accept move (hill climbing)
            current_score = new_score
            current_set = new_set
            current_lines = list(current_set)
            current_points = new_points
            remaining_lines = list(pool_set - current_set)
        else:
            # Revert (implicitly, we just don't update current_set)
            pass
            
        steps += 1
        if steps % 100 == 0:
            # Optional: Restart from best if stuck?
            pass
            
    # Convert best_points to array
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Verify shapes
    if points_arr.shape != (54, 2):
        # Pad with dummy points if not enough? 
        # But we want max score. If we don't have 54, we have a problem.
        # With 27 lines, we should have many intersections.
        # If not, pad with random integer points.
        while points_arr.shape[0] < 54:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in [tuple(pt) for pt in points_arr]:
                points_arr = np.vstack([points_arr, np.array(p)])
                
    if lines_arr.shape != (27, 3):
        # Should not happen
        pass
        
    return points_arr, lines_arr, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a line-based constructive strategy.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    # Generate pool of candidate lines
    pool = _generate_line_pool(max_coord=15)
    
    # Run the line-based search
    points, lines, score = _find_best_lines_and_points(pool, 27, 54, timeout - 10)
    
    # Final validation
    points = points.astype(np.int64)
    lines = lines.astype(np.int64)
    
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    
    # Check duplicates
    pts_set = set([tuple(p) for p in points])
    if len(pts_set) < 54:
        # Fallback to a known valid configuration if duplicates occurred
        # This shouldn't happen with the set logic, but just in case
        pass
        
    lines_set = set([tuple(l) for l in lines])
    if len(lines_set) < 27:
        pass
        
    return points, lines

# EVOLVE-BLOCK-END
