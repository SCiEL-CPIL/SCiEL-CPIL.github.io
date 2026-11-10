import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation A*x + B*y + C = 0 to a unique representative."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    
    A, B, C = A // g, B // g, C // g
    
    # Ensure the first non-zero coefficient is positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B == 0 and C < 0:
        A, B, C = -A, -B, -C
        
    return (A, B, C)

def _count_incidence(points, lines):
    """
    Count incidences between points (N,2) and lines (M,3).
    Returns an integer count.
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    # Vectorized computation: (M, 1) * (1, N) + (M, 1) * (1, N) + (M, 1)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _generate_initial_grid(k, m, shear=0, scale=1, ox=0, oy=0):
    """
    Generate a k x m grid with optional shear and scaling.
    Returns (N, 2) array.
    """
    xs = np.arange(k) * scale
    ys = np.arange(m) * scale
    px, py = np.meshgrid(xs, ys, indexing='ij')
    
    # Apply shear: x' = x + shear * y
    px = px + shear * py
    
    # Translate
    px = px + ox
    py = py + oy
    
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _build_candidate_pool(points, max_candidates=2000):
    """
    Build a pool of candidate lines from:
    1. Lines through pairs of points.
    2. Horizontal, Vertical, and Diagonal lines with small coefficients.
    """
    n = len(points)
    candidates = set()
    
    # 1. Lines through pairs
    # To avoid O(N^2) explosion if N is large, we sample if necessary, but N=54 is small.
    for i in range(n):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            candidates.add(key)
            
    # 2. Structured lines
    # Collect unique x, y coordinates to find rows/cols
    xs = set(int(p[0]) for p in points)
    ys = set(int(p[1]) for p in points)
    
    for x in xs:
        candidates.add(_normalize_line(1, 0, -x)) # Vertical x = c
    for y in ys:
        candidates.add(_normalize_line(0, 1, -y)) # Horizontal y = c
        
    # Diagonals y = x + c and y = -x + c
    diffs = set(int(p[1]) - int(p[0]) for p in points)
    sums = set(int(p[1]) + int(p[0]) for p in points)
    for d in diffs:
        candidates.add(_normalize_line(1, -1, -d)) # y - x = d => x - y - d = 0
    for s in sums:
        candidates.add(_normalize_line(1, 1, -s))   # y + x = s => x + y - s = 0
        
    # Small coefficient lines to capture near-misses or different structures
    # A, B in [-3, 3], C in range of points
    min_x = min(xs) if xs else -10
    max_x = max(xs) if xs else 10
    min_y = min(ys) if ys else -10
    max_y = max(ys) if ys else 10
    
    c_min = min(min_x, min_y) - 10
    c_max = max(max_x, max_y) + 10
    
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            # Sample C values to avoid generating too many
            # We care about lines that might pass through points.
            # If A != 0, C = -(A*x + B*y).
            # Just add a few generic ones
            for C in range(c_min, c_max + 1, 1):
                 # This might be too many. Let's restrict C based on point bounds.
                 # Actually, for a specific (A,B), the relevant C values are determined by points.
                 pass
            
    # More targeted: For small A,B, add lines passing through each point
    for A in range(-2, 3):
        for B in range(-2, 3):
            if A == 0 and B == 0:
                continue
            if A == 0 and B == 0: continue
            # Skip if A,B are large or zero in a way that doesn't help
            for p in points:
                x, y = int(p[0]), int(p[1])
                C = -(A * x + B * y)
                candidates.add(_normalize_line(A, B, C))
                
    # Limit pool size if it's huge (unlikely for 54 points, but safe)
    if len(candidates) > max_candidates:
        # Prioritize lines with higher incidence? We don't know incidence yet.
        # Just take a random subset or the first max_candidates.
        # Better: return all, let the selection step handle it.
        pass
        
    return list(candidates)

def _select_best_lines(points, candidate_lines, k=27):
    """
    Select the k lines from candidate_lines that have the highest incidence with points.
    candidate_lines: list of (A, B, C) tuples.
    """
    if not candidate_lines:
        return np.zeros((k, 3), dtype=np.int64)
        
    lines_arr = np.array(candidate_lines, dtype=np.int64)
    counts = np.zeros(len(lines_arr), dtype=np.int64)
    
    px = points[:, 0]
    py = points[:, 1]
    
    # Process in batches if lines_arr is very large to save memory, 
    # but for ~1000-2000 lines and 54 points, it's small.
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k indices
    # np.argsort returns indices that would sort the array.
    # We want descending order of counts.
    top_k_indices = np.argsort(counts)[::-1][:k]
    
    selected_lines = lines_arr[top_k_indices]
    
    # Pad if necessary
    if len(selected_lines) < k:
        pad = np.zeros((k - len(selected_lines), 3), dtype=np.int64)
        # Use dummy lines that won't be valid or will be replaced
        pad[:, 1] = 1
        pad[:, 2] = -10000
        selected_lines = np.vstack([selected_lines, pad])
        
    return selected_lines

def _local_search(points, lines, time_limit, seed):
    """
    Perform local search on points and lines.
    Returns (points, lines, score).
    """
    random.seed(seed)
    points = points.copy()
    lines = lines.copy()
    current_score = _count_incidence(points, lines)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 100
    pool = _build_candidate_pool(points)
    pool_set = set(pool)
    
    # Pre-compute incidence matrix for current lines? 
    # It's easier to just recompute score when needed, it's fast enough.
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
            
        move_type = random.random()
        improved = False
        
        if move_type < 0.7:
            # Move a point
            i = random.randint(0, len(points) - 1)
            old_x, old_y = int(points[i, 0]), int(points[i, 1])
            
            # Try random moves
            for _ in range(10):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Check uniqueness
                # This is O(N). For N=54, it's fine.
                is_unique = True
                for j in range(len(points)):
                    if j == i: continue
                    if points[j, 0] == new_x and points[j, 1] == new_y:
                        is_unique = False
                        break
                if not is_unique:
                    continue
                
                # Apply move
                points[i, 0] = new_x
                points[i, 1] = new_y
                
                # Update pool with lines through new point
                # We can just rebuild the pool or add specific lines.
                # Rebuilding is O(N^2) = 2700 ops, which is fast.
                # Let's rebuild to ensure we have all relevant lines.
                pool = _build_candidate_pool(points)
                
                # Select best lines
                new_lines = _select_best_lines(points, pool, 27)
                new_score = _count_incidence(points, new_lines)
                
                if new_score > current_score:
                    lines = new_lines
                    current_score = new_score
                    improved = True
                    no_improve = 0
                    break
                elif new_score == current_score and random.random() < 0.1:
                    lines = new_lines
                    no_improve += 1
                else:
                    # Revert
                    points[i, 0] = old_x
                    points[i, 1] = old_y
                    
        else:
            # Swap a line
            # Pick a line index to replace
            i = random.randint(0, 26)
            old_line = lines[i]
            
            # Find best line from pool that is NOT in current lines
            current_line_set = set(map(tuple, lines.tolist()))
            
            # To speed this up, we can pre-compute counts for the whole pool
            # and pick the best one not in current_line_set.
            lines_arr = np.array(pool, dtype=np.int64)
            if len(lines_arr) == 0:
                continue
                
            px = points[:, 0]
            py = points[:, 1]
            A = lines_arr[:, 0]
            B = lines_arr[:, 1]
            C = lines_arr[:, 2]
            vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
            counts = np.sum(vals == 0, axis=1)
            
            # Mask out lines already in the set
            mask = np.ones(len(lines_arr), dtype=bool)
            for idx, line in enumerate(pool):
                if line in current_line_set:
                    mask[idx] = False
                    
            if not np.any(mask):
                continue
                
            valid_indices = np.where(mask)[0]
            valid_counts = counts[valid_indices]
            best_idx_in_valid = valid_indices[np.argmax(valid_counts)]
            best_line = pool[best_idx_in_valid]
            
            if best_line == tuple(old_line):
                continue
                
            # Apply swap
            lines[i] = best_line
            new_score = _count_incidence(points, lines)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                no_improve = 0
            else:
                # Revert
                lines[i] = old_line
                
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return points, lines, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # List of initial configurations to try
    initial_configs = []
    
    # Standard grids
    for k in [1, 2, 3, 6, 9, 18, 27, 54]:
        if 54 % k == 0:
            m = 54 // k
            initial_configs.append(_generate_initial_grid(k, m))
            
    # Sheared grids
    for k in [3, 6, 9]:
        if 54 % k == 0:
            m = 54 // k
            for shear in [-2, -1, 1, 2]:
                initial_configs.append(_generate_initial_grid(k, m, shear=shear))
                
    # Random points in a box
    for _ in range(5):
        pts = np.random.randint(-10, 10, size=(54, 2))
        # Ensure uniqueness
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) < 54:
            # Pad
            while len(unique_pts) < 54:
                new_pt = np.array([random.randint(-20, 20), random.randint(-20, 20)])
                if not np.any((unique_pts == new_pt).all(axis=1)):
                    unique_pts = np.vstack([unique_pts, new_pt[None, :]])
            pts = unique_pts[:54]
        initial_configs.append(pts.astype(np.int64))
        
    # Evaluate each initial configuration
    for i, pts in enumerate(initial_configs):
        if time.time() - start_time > deadline - 60:
            break
            
        # Build pool
        pool = _build_candidate_pool(pts)
        
        # Select best lines
        lines = _select_best_lines(pts, pool, 27)
        
        # Calculate score
        score = _count_incidence(pts, lines)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
            
    # Local search from the best configuration found
    if best_points is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 30:
            # Run local search with a few different seeds to explore neighborhood
            for seed in [42, 123, 999]:
                if time.time() - start_time > deadline - 10:
                    break
                alloc_time = min(remaining_time - 10, 120)
                p, l, s = _local_search(best_points, best_lines, alloc_time, seed)
                if s > best_score:
                    best_score = s
                    best_points = p.copy()
                    best_lines = l.copy()
                
                # Re-evaluate remaining time
                remaining_time = deadline - time.time()
                if remaining_time < 30:
                    break

    # Fallback
    if best_points is None:
        best_points = _generate_initial_grid(6, 9)
        pool = _build_candidate_pool(best_points)
        best_lines = _select_best_lines(best_points, pool, 27)
        best_score = _count_incidence(best_points, best_lines)
        
    # Final validation and formatting
    final_points = np.array(best_points, dtype=np.int64)
    final_lines = np.array(best_lines, dtype=np.int64)
    
    # Ensure shapes
    if final_points.shape != (54, 2):
        final_points = _generate_initial_grid(6, 9)
        pool = _build_candidate_pool(final_points)
        final_lines = _select_best_lines(final_points, pool, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -10000
            final_lines = np.vstack([final_lines, pad])
            
    # Check for duplicate points
    unique_pts = np.unique(final_points, axis=0)
    if len(unique_pts) < 54:
        # This shouldn't happen if we were careful, but fix it just in case
        added = 0
        while len(unique_pts) < 54:
            pt = np.array([100 + added, 100 + added], dtype=np.int64)
            if not np.any((unique_pts == pt).all(axis=1)):
                unique_pts = np.vstack([unique_pts, pt[None, :]])
            added += 1
        final_points = unique_pts[:54]
        pool = _build_candidate_pool(final_points)
        final_lines = _select_best_lines(final_points, pool, 27)
        
    # Check for invalid lines (A=B=0)
    for i in range(27):
        if final_lines[i, 0] == 0 and final_lines[i, 1] == 0:
            final_lines[i, 1] = 1
            final_lines[i, 2] = -10000 - i

    return final_points.astype(np.int64), final_lines.astype(np.int64)
