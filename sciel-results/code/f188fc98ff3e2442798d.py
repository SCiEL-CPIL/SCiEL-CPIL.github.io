import numpy as np
import time
from math import gcd
from itertools import product
import random

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

def _count_incidences(points, lines):
    """
    Count incidences between points and lines.
    points: (N, 2) array
    lines: (M, 3) array
    Returns: int
    """
    # A*x + B*y + C == 0
    # points: (N, 2) -> (N, 1, 2)
    # lines: (M, 3) -> (1, M, 3)
    # We need to compute A*x + B*y + C for all pairs.
    
    # To avoid memory explosion for large N, M, we can do it in chunks or use broadcasting carefully.
    # N=54, M=27 is small enough for full broadcasting: 54*27 = 1458 values.
    
    # points[:, 0] is (N,), lines[:, 0] is (M,)
    # We want a (N, M) matrix of values.
    
    x = points[:, 0].astype(np.int64)
    y = points[:, 1].astype(np.int64)
    
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    
    # (N, 1) * (1, M) + (N, 1) * (1, M) + (1, 1) * (1, M) -> (N, M)
    # val[i, j] = A[j]*x[i] + B[j]*y[i] + C[j]
    
    vals = np.outer(x, A) + np.outer(y, B) + C[np.newaxis, :]
    
    return np.sum(vals == 0)

def _generate_line_pool(points, max_coeff=20):
    """
    Generate a pool of candidate lines based on the points.
    Includes vertical, horizontal, and lines with small integer coefficients.
    """
    lines_set = set()
    
    # Vertical and Horizontal
    for x in np.unique(points[:, 0]):
        lines_set.add(_normalize_line(1, 0, -x))
    for y in np.unique(points[:, 1]):
        lines_set.add(_normalize_line(0, 1, -y))
    
    # Lines defined by pairs of points
    # This might be too many, so we limit or use a heuristic.
    # Actually, for 54 points, pairs are ~1400. That's a manageable pool size.
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            norm_line = _normalize_line(A, B, C)
            if norm_line != (0, 0, 0):
                # Only keep lines with small coefficients to keep the pool diverse and "simple"
                if max(abs(norm_line[0]), abs(norm_line[1]), abs(norm_line[2])) <= max_coeff:
                    lines_set.add(norm_line)
                    
    # Also add some random simple lines
    for A in range(-max_coeff, max_coeff+1):
        for B in range(-max_coeff, max_coeff+1):
            if A == 0 and B == 0:
                continue
            # C should be such that the line passes near the center of the point cloud
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])
            C = -int(round(A*cx + B*cy))
            norm_line = _normalize_line(A, B, C)
            if max(abs(norm_line[0]), abs(norm_line[1]), abs(norm_line[2])) <= max_coeff:
                lines_set.add(norm_line)
                
    return list(lines_set)

def _evaluate_line_set(points, lines):
    """
    Evaluate a set of lines for given points.
    Returns the number of incidences.
    """
    lines_arr = np.array(lines, dtype=np.int64)
    return _count_incidences(points, lines_arr)

def _initial_grid_config():
    """
    Generate the initial 6x9 grid configuration.
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
            
    lines = []
    # 6 Vertical
    for i in range(6):
        lines.append(_normalize_line(1, 0, -i))
    # 9 Horizontal
    for j in range(9):
        lines.append(_normalize_line(0, 1, -j))
    # 6 Slope 1
    optimal_c_slope1 = [4, 5, 6, 7, 8, 9]
    for c in optimal_c_slope1:
        lines.append(_normalize_line(1, 1, -c))
    # 6 Slope -1
    optimal_c_slope_minus1 = [-4, -3, -2, -1, 0, 1]
    for c in optimal_c_slope_minus1:
        lines.append(_normalize_line(1, -1, -c))
        
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    return points_arr, lines_arr

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580.0  # 20s margin
    
    # Start with the known good configuration
    points, lines = _initial_grid_config()
    best_inc = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Generate initial pool
    pool = _generate_line_pool(points, max_coeff=30)
    pool_set = set(pool)
    
    # Local Search
    # We will try to swap lines to improve incidence count.
    # We will also try to move points slightly.
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Try a random restart or larger perturbation
            # For now, just break and try a different strategy if time permits
            # Or just restart the search with a shuffled pool order
            no_improvement_count = 0
            # Shuffle pool to try different swaps
            random.shuffle(pool)
            continue
            
        improved = False
        
        # Try swapping each line with a better one from the pool
        # To speed up, we can calculate the marginal gain/loss.
        # Current Inc = I
        # If we remove line L and add line L', new Inc = I - count(L) + count(L') - (interaction terms)
        # Interaction terms: points on both L and L' are counted in I, but in new set, if point is on L', it's counted once.
        # Actually, Inc = sum_{l in lines} count(l) - sum_{l1, l2} count(l1 & l2) + ...
        # This is complex. Direct evaluation is fast enough for 27 lines.
        
        # Optimization: Precompute counts for all lines in pool?
        # Pool size might be ~1000-2000. 27 lines in current set.
        # Evaluating 1 candidate line against 54 points is very fast.
        
        # Let's precompute the incidence count for every line in the pool against the current points.
        # This allows us to quickly find the best replacement for any line.
        
        # However, the "value" of a line depends on the other lines (due to double counting in incidence sum? No, incidence is sum of individual line incidences).
        # Wait, I = sum_{p, l} [p in l]. This is simply the sum of the number of points on each line.
        # There is no "interaction" penalty in the definition of I.
        # I = sum_{l in lines} (number of points on l).
        # So, to maximize I, we just need to pick the 27 lines from the pool that have the highest individual point counts, subject to no duplicate lines.
        
        # If this is true, the problem is simply:
        # 1. Fix points.
        # 2. Generate pool of lines.
        # 3. Count points on each line in pool.
        # 4. Pick top 27 distinct lines.
        
        # Let's verify this logic.
        # "I is the number of (point, line) pairs with A*x + B*y + C == 0"
        # Yes, it is a sum over pairs. Linearity holds.
        # So, for a fixed set of points, the optimal set of lines is just the top-K lines by point count.
        
        # This simplifies the search significantly!
        # We only need to search for the optimal POINTS.
        # For any given points, the optimal lines are the top 27 from the pool.
        
        # So the algorithm is:
        # 1. Start with initial points.
        # 2. Generate pool.
        # 3. Find top 27 lines.
        # 4. Calculate I.
        # 5. Try to move one point to a nearby location.
        # 6. If I improves, accept. Else, reject.
        # 7. Repeat.
        
        # Let's re-architect the loop based on this insight.
        
        # We need a fast way to update line counts when a point moves.
        # Or just recompute. Recomputing 2000 lines * 54 points is 108k ops. Very fast.
        
        # Let's implement the point search.
        
        # Current state
        current_inc = best_inc
        
        # Generate pool for current points
        pool = _generate_line_pool(points, max_coeff=30)
        
        # Calculate scores for all lines in pool
        # scores: dict line -> count
        line_scores = {}
        for l in pool:
            # l is tuple (A, B, C)
            # Count points on l
            # We can vectorize this for the whole pool at once?
            # Pool size ~2000. Points 54.
            # 2000 * 54 = 108k operations.
            pass
        
        # Let's do it vectorized for the whole pool.
        pool_arr = np.array(pool, dtype=np.int64)
        # (P, 3)
        
        # Points (54, 2)
        x = points[:, 0].astype(np.int64)
        y = points[:, 1].astype(np.int64)
        
        A = pool_arr[:, 0]
        B = pool_arr[:, 1]
        C = pool_arr[:, 2]
        
        # (54, 1) * (1, P) -> (54, P)
        # val[i, j] = A[j]*x[i] + B[j]*y[i] + C[j]
        vals = np.outer(x, A) + np.outer(y, B) + C[np.newaxis, :]
        
        # Count zeros for each line (column)
        line_counts = np.sum(vals == 0, axis=0)
        
        # Find top 27
        # We need distinct lines. The pool is already distinct (set).
        # So just pick the top 27 indices.
        
        top_27_indices = np.argsort(line_counts)[-27:]
        current_lines = pool[top_27_indices]
        current_inc = np.sum(line_counts[top_27_indices])
        
        if current_inc > best_inc:
            best_inc = current_inc
            best_points = points.copy()
            best_lines = np.array(current_lines, dtype=np.int64)
            no_improvement_count = 0
            improved = True
        else:
            no_improvement_count += 1
            
        # Try moving a point
        # Pick a random point
        idx = random.randint(0, len(points)-1)
        old_x, old_y = points[idx]
        
        # Try nearby moves
        moved = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Check if new point is duplicate
                # For simplicity, assume grid is sparse enough or check
                # We can just try it and if duplicate, skip
                # To be safe, check if (new_x, new_y) is in points
                # This is O(N). N=54. Fast.
                is_dup = False
                for k in range(len(points)):
                    if k == idx:
                        continue
                    if points[k][0] == new_x and points[k][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Move point
                points[idx] = [new_x, new_y]
                
                # Evaluate new configuration
                # Regenerate pool? The pool depends on points.
                # If we move a point, the "lines defined by pairs" change.
                # However, the "simple lines" (vertical, horizontal, small coeff) remain valid candidates.
                # The pair-defined lines might change, but the pool of "simple" lines is stable.
                # Let's use a static pool of simple lines (vertical, horizontal, small A,B) 
                # plus the lines defined by the CURRENT points.
                # Actually, just using the simple lines pool is a good approximation.
                # Let's stick to the simple lines pool for speed, or update the pool.
                # Updating pool is O(N^2). 54^2 = 2916. Fast.
                
                new_pool = _generate_line_pool(points, max_coeff=30)
                new_pool_arr = np.array(new_pool, dtype=np.int64)
                
                x_new = points[:, 0].astype(np.int64)
                y_new = points[:, 1].astype(np.int64)
                
                A_new = new_pool_arr[:, 0]
                B_new = new_pool_arr[:, 1]
                C_new = new_pool_arr[:, 2]
                
                vals_new = np.outer(x_new, A_new) + np.outer(y_new, B_new) + C_new[np.newaxis, :]
                line_counts_new = np.sum(vals_new == 0, axis=0)
                
                top_27_new_indices = np.argsort(line_counts_new)[-27:]
                new_inc = np.sum(line_counts_new[top_27_new_indices])
                
                if new_inc > current_inc:
                    current_inc = new_inc
                    if new_inc > best_inc:
                        best_inc = new_inc
                        best_points = points.copy()
                        best_lines = np.array(new_pool[top_27_new_indices], dtype=np.int64)
                        no_improvement_count = 0
                    moved = True
                    break
                else:
                    # Revert move
                    points[idx] = [old_x, old_y]
                    
            if moved:
                break
                
        if not moved and no_improvement_count > 200:
            # Try a larger jump or different start?
            # For now, just continue.
            pass
            
    # Return best found
    return best_points, best_lines

# Quick sanity check and run
if __name__ == "__main__":
    pts, lns = szemeredi_trotter_incidences()
    print(pts.shape, lns.shape)
    print(_count_incidences(pts, lns))
