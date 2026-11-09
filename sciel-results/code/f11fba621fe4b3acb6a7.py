import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(gcd(abs(int(A)), abs(int(B))), abs(int(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    # Make first nonzero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64).T
    B = lines[:, 1:2].astype(np.int64).T
    C = lines[:, 2:3].astype(np.int64).T
    total = px * A + py * B + C
    return int(np.sum(total == 0))

def _generate_grid_points(w, h, remove_frac=0.0, seed=42):
    """Generate a grid of w*h points, optionally removing some randomly."""
    rng = np.random.RandomState(seed)
    xs = np.arange(1, w + 1)
    ys = np.arange(1, h + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if remove_frac > 0:
        n_total = len(points)
        n_remove = int(n_total * remove_frac)
        if n_remove > 0:
            indices = rng.choice(n_total, n_remove, replace=False)
            mask = np.ones(n_total, dtype=bool)
            mask[indices] = False
            points = points[mask]
            
    # We need exactly 54 points. If we have more, trim. If less, pad (shouldn't happen with small grids).
    if len(points) > 54:
        indices = rng.choice(len(points), 54, replace=False)
        points = points[indices]
    elif len(points) < 54:
        # Pad with points far away? No, that hurts. 
        # Better to only use grids that yield >= 54 points or exactly 54.
        # For this strategy, we'll stick to grids that are exactly 54 or we sample from a larger grid.
        pass
        
    return points

def _generate_candidate_lines(w, h, max_slope=4):
    """Generate a diverse set of candidate lines for a w x h grid."""
    candidates = set()
    
    # Horizontal
    for c in range(1, h + 1):
        candidates.add(_normalize_line(0, 1, -c))
    
    # Vertical
    for c in range(1, w + 1):
        candidates.add(_normalize_line(1, 0, -c))
    
    # Diagonals and other slopes
    for A in range(-max_slope, max_slope + 1):
        for B in range(-max_slope, max_slope + 1):
            if A == 0 and B == 0:
                continue
            # Normalize direction
            g = gcd(abs(A), abs(B))
            if g > 0:
                A, B = A // g, B // g
            if A < 0 or (A == 0 and B < 0):
                A, B = -A, -B
            
            # Determine range of C
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            # x in [1, w], y in [1, h]
            min_val = A * 1 + B * 1
            max_val = A * w + B * h
            # Since A, B can be negative, we need to be careful with min/max
            vals = [A*1 + B*1, A*1 + B*h, A*w + B*1, A*w + B*h]
            min_sum = min(vals)
            max_sum = max(vals)
            
            for C in range(-max_sum, -min_sum + 1):
                candidates.add(_normalize_line(A, B, C))
                
    # Add some "random" lines that might pass through clusters
    # This is heuristic, but the SA will handle it.
    
    return list(candidates)

def _simulated_annealing(points, initial_lines, all_candidates, time_limit, start_time):
    """
    Simulated Annealing to improve line selection.
    """
    n_lines = 27
    current_lines = list(initial_lines)
    current_score = _incidence_count(points, np.array(current_lines, dtype=np.int64))
    
    # Precompute incidence counts for all candidates
    cand_arr = np.array(all_candidates, dtype=np.int64)
    inc_counts = _compute_all_incidences(points, cand_arr)
    
    # Map line tuple to index
    line_to_idx = {line: i for i, line in enumerate(all_candidates)}
    
    # Current line indices
    current_indices = [line_to_idx[line] for line in current_lines]
    current_set = set(current_indices)
    
    best_lines = list(current_lines)
    best_score = current_score
    
    T = 10.0
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    while time.time() < start_time + time_limit:
        if T < 0.01:
            break
            
        # Pick a random line to swap out
        i = rng.randrange(n_lines)
        old_idx = current_indices[i]
        
        # Pick a random line to swap in (not currently selected)
        # To speed up, we can pick from a subset of high-incidence lines or random
        # Let's pick a random candidate index
        new_idx = rng.randrange(len(all_candidates))
        
        if new_idx in current_set:
            continue
            
        # Calculate delta
        old_inc = inc_counts[old_idx]
        new_inc = inc_counts[new_idx]
        delta = new_inc - old_inc
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            current_set.remove(old_idx)
            current_set.add(new_idx)
            current_indices[i] = new_idx
            current_lines[i] = all_candidates[new_idx]
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_lines = list(current_lines)
                
        T *= cooling_rate
        
    return best_lines, best_score

def _compute_all_incidences(points, candidates):
    """Compute incidence counts for all candidate lines against points."""
    n = len(points)
    m = len(candidates)
    if n == 0 or m == 0:
        return np.zeros(m, dtype=np.int64)
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = candidates[:, 0:1].astype(np.int64).T
    B = candidates[:, 1:2].astype(np.int64).T
    C = candidates[:, 2:3].astype(np.int64).T
    total = px * A + py * B + C
    return np.sum(total == 0, axis=0)

def _local_search_points(points, selected_lines, time_limit, start_time):
    """
    Local search on points: try moving points to improve incidences.
    This is tricky because moving a point changes incidences for all lines.
    We'll do a simple hill-climbing: move one point at a time to a nearby integer coord.
    """
    n_pts = len(points)
    lines_arr = np.array(selected_lines, dtype=np.int64)
    current_score = _incidence_count(points, lines_arr)
    
    # Precompute line coefficients
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    best_points = points.copy()
    improved = True
    while improved and time.time() < start_time + time_limit:
        improved = False
        best_gain = 0
        best_move = None
        
        # Sample a subset of points if too many? No, 54 is small.
        # But checking all neighbors for all points is 54 * 25 * 27 ops ~ 36k ops. Very fast.
        for i in range(n_pts):
            x, y = best_points[i]
            old_incs = np.sum(A * x + B * y + C == 0)
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 1 or ny < 1:
                        continue
                    
                    # Check duplicate
                    # This is O(N) per check, so 54*25*54 ~ 72k ops. Acceptable.
                    is_dup = False
                    for j in range(n_pts):
                        if i == j:
                            continue
                        if best_points[j][0] == nx and best_points[j][1] == ny:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_incs = np.sum(A * nx + B * ny + C == 0)
                    gain = new_incs - old_incs
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, nx, ny)
        
        if best_move is not None:
            i, nx, ny = best_move
            best_points[i] = [nx, ny]
            current_score += best_gain
            improved = True
            
    return best_points, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try several grid configurations
    # 1. 6x9, 9x6, 3x18, 18x3, 2x27, 27x2
    # 2. Subsets of larger grids? e.g., sample 54 from 10x10?
    #    The current program uses fixed grids. Let's stick to standard grids first but improve the search.
    
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        # Maybe 54 = 54*1?
        (54, 1),
        (1, 54),
    ]
    
    for w, h in grid_configs:
        if time.time() > deadline - 60:
            break
            
        # Generate points
        points = _generate_grid_points(w, h, remove_frac=0.0, seed=42)
        if len(points) != 54:
            continue # Skip if we didn't get exactly 54 (shouldn't happen for these dims)
            
        # Generate candidates
        candidates = _generate_candidate_lines(w, h, max_slope=5)
        if len(candidates) < 27:
            continue
            
        # Initial selection: Top 27 by incidence
        cand_arr = np.array(candidates, dtype=np.int64)
        inc_counts = _compute_all_incidences(points, cand_arr)
        
        # Get top 27 indices
        top_indices = np.argsort(inc_counts)[::-1][:27]
        initial_lines = [candidates[i] for i in top_indices]
        
        # Run Simulated Annealing
        remaining_time = max(10, deadline - time.time() - 30)
        sa_time = remaining_time * 0.6
        ls_time = remaining_time * 0.4
        
        sa_lines, sa_score = _simulated_annealing(points, initial_lines, candidates, sa_time, start_time)
        
        # Run Local Search on Points
        # Note: Moving points might break the "grid" structure but could find better local optima.
        # However, the task implies we can choose ANY 54 points.
        # So we should definitely try to optimize points too.
        
        final_lines = np.array(sa_lines, dtype=np.int64)
        final_points, final_score = _local_search_points(points, sa_lines, ls_time, start_time)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()
            best_lines = final_lines.copy()
            
    # Fallback
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
        
    return best_points, best_lines
