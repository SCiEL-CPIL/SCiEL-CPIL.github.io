import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    # Make the first nonzero entry positive
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _count_inc(points, lines):
    """
    Count incidences between points (N, 2) and lines (M, 3).
    Returns an integer count.
    """
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    res = A * x + B * y + C
    return int(np.sum(res == 0))

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0  # Safety margin for the 630s kill limit
    
    # Grid dimensions
    # The current program tries (8,7) and (7,8).
    # Inspiration 1 uses (7,8) and iterates over removals.
    # We will use (7,8) as the primary grid since it yields 56 points, 
    # and we need to remove 2. This is a denser structure for low-slope lines.
    
    nx, ny = 7, 8
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points) # 56
    
    # Generate candidate lines
    # We want a rich set of candidates to choose the top 27 from.
    # The current program only generated axis-aligned and diagonal (slope +/-1).
    # Inspiration 1 generated more slopes.
    # Let's generate all lines passing through at least 2 points of the grid.
    # This is the most robust way to ensure we have the "best" lines available.
    
    candidates_set = set()
    
    # 1. Axis aligned
    for c in range(nx):
        candidates_set.add((1, 0, -c))
    for c in range(ny):
        candidates_set.add((0, 1, -c))
        
    # 2. General lines
    # A line is defined by two distinct points (x1, y1) and (x2, y2).
    # Equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    # A = y2-y1, B = -(x2-x1), C = (x2-x1)y1 - (y2-y1)x1
    # We iterate over all pairs of points in the grid.
    # There are 56 points, so ~1500 pairs. This is fast.
    
    for i in range(n_pts):
        x1, y1 = base_points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = base_points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            # Normalize to handle duplicates and scaling
            norm = _normalize_line(A, B, C)
            if norm is not None:
                candidates_set.add(norm)
                
    # Convert to array
    unique_lines = np.array(list(candidates_set), dtype=np.int64)
    n_lines = len(unique_lines)
    
    # Precompute incidence matrix: M[i, j] = 1 if point i is on line j
    # Points: (56, 2), Lines: (N, 3)
    A_arr = unique_lines[:, 0:1]
    B_arr = unique_lines[:, 1:2]
    C_arr = unique_lines[:, 2:3]
    x_all = base_points[:, 0:1].T
    y_all = base_points[:, 1:1+1].T # (1, 56)
    
    # res: (N_lines, 1) * (1, 56) + ... -> (N_lines, 56)
    vals = A_arr * x_all + B_arr * y_all + C_arr
    incidence = (vals == 0).astype(np.int8)
    
    # total_inc[j] = number of points on line j
    total_inc = incidence.sum(axis=0)
    
    best_score = -1
    best_points = None
    best_lines = None
    best_removed = None
    
    # Strategy: Select 54 points (remove 2) and 27 lines to maximize incidences.
    # We can iterate over the pair of points to remove (56 choose 2 = 1540 pairs).
    # For each pair, we calculate the incidence of each line with the remaining 54 points.
    # Then we pick the top 27 lines.
    # This is O(1540 * N_lines). If N_lines is ~2000, that's ~3M ops, very fast.
    
    # Optimization:
    # line_inc_after_removal[j] = total_inc[j] - incidence[removed_1, j] - incidence[removed_2, j]
    # We want to maximize sum(top_27(line_inc_after_removal))
    
    # Let's iterate over all pairs of points to remove.
    for i, j in combinations(range(n_pts), 2):
        if time.time() - start_time > deadline - 5:
            break
            
        # Calculate incidence counts for all lines after removing points i and j
        # incidence[i] and incidence[j] are vectors of length N_lines
        line_inc = total_inc - incidence[i] - incidence[j]
        
        # Get top 27 indices
        # np.argsort is O(N log N). N is ~2000. 1540 * 2000 log 2000 is ~ 1540 * 2000 * 11 ~ 33M ops.
        # This might be slightly slow in Python if not vectorized well, but NumPy is C.
        # Let's check if we can be smarter or if this is fast enough.
        # 33M ops in NumPy should take < 0.1s.
        
        top_indices = np.argsort(-line_inc)[:27]
        score = int(line_inc[top_indices].sum())
        
        if score > best_score:
            best_score = score
            # Prepare points: all points except i and j
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False
            mask[j] = False
            best_points = base_points[mask].copy()
            best_lines = unique_lines[top_indices].copy()
            best_removed = (i, j)
            
    # Fallback if loop didn't run or something went wrong
    if best_points is None:
        best_points = base_points[:54]
        best_removed = (54, 55)
        line_inc = total_inc - incidence[54] - incidence[55]
        top_indices = np.argsort(-line_inc)[:27]
        best_lines = unique_lines[top_indices].copy()
        best_score = int(line_inc[top_indices].sum())
        
    # Refinement: Local search
    # Can we improve by swapping one removed point with one kept point?
    # Or swapping lines?
    # The line selection is optimal for a fixed set of points (greedy top-k is optimal for sum of top-k).
    # So we only need to refine the point set.
    
    if best_removed is not None and time.time() - start_time < deadline - 5:
        removed_set = set(best_removed)
        kept_indices = [k for k in range(n_pts) if k not in removed_set]
        current_score = best_score
        improved = True
        
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            # Try swapping each removed point with each kept point
            # There are 2 removed points and 54 kept points. 108 swaps to check.
            for r in list(removed_set):
                if time.time() - start_time > deadline - 5:
                    break
                for k in kept_indices:
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    # New removed set: replace r with k
                    # Other removed point:
                    other = best_removed[0] if best_removed[1] == r else best_removed[1]
                    
                    # New line incidences
                    # line_inc = total_inc - incidence[other] - incidence[k]
                    line_inc = total_inc - incidence[other] - incidence[k]
                    top_indices = np.argsort(-line_inc)[:27]
                    new_score = int(line_inc[top_indices].sum())
                    
                    if new_score > current_score:
                        current_score = new_score
                        # Update state
                        new_removed = (other, k)
                        mask = np.ones(n_pts, dtype=bool)
                        mask[other] = False
                        mask[k] = False
                        best_points = base_points[mask].copy()
                        best_lines = unique_lines[top_indices].copy()
                        best_removed = new_removed
                        removed_set = set(new_removed)
                        kept_indices = [x for x in range(n_pts) if x not in removed_set]
                        improved = True
                        break # Restart the inner loop with new state? Or just continue? 
                        # Breaking the inner loop is safer to ensure consistency of kept_indices list if we update it.
                        # But we updated kept_indices above.
        
        best_score = current_score

    return best_points, best_lines
