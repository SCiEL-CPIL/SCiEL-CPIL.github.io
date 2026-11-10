# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _canon_line(A, B, C):
    """Canonical form: divide by gcd, make first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    # points: (N, 2), lines: (M, 3)
    # Vectorized check
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    
    count = 0
    for j in range(len(lines)):
        vals = A[j] * px + B[j] * py + C[j]
        count += np.sum(vals == 0)
    return int(count)

def _generate_grid_points(rows, cols):
    """Generate a grid of points."""
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing='xy')
    points = np.column_stack((px.ravel(), py.ravel()))
    return points.astype(np.int64)

def _generate_line_pool(points, max_lines=5000):
    """
    Generate a pool of candidate lines.
    Includes all lines passing through at least 2 points.
    """
    n = len(points)
    lines_set = set()
    
    # Lines through pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            c = _canon_line(A, B, C)
            lines_set.add(c)
            
            if len(lines_set) > max_lines:
                break
        if len(lines_set) > max_lines:
            break
            
    # Add some random lines to explore space
    for _ in range(200):
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        if A == 0 and B == 0:
            continue
        C = random.randint(-50, 50)
        c = _canon_line(A, B, C)
        lines_set.add(c)
        
    return list(lines_set)

def _greedy_select_lines(points, pool, k=27):
    """
    Greedy selection of k lines that maximize incidences.
    """
    pool = list(set(pool))
    random.shuffle(pool)
    
    # Precompute incidence counts for all lines in pool
    # To speed up, we can just evaluate all
    counts = {}
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    
    for line in pool:
        A, B, C = line
        vals = A * px + B * py + C
        counts[line] = int(np.sum(vals == 0))
        
    # Sort by count
    sorted_lines = sorted(counts.keys(), key=lambda l: counts[l], reverse=True)
    
    selected = []
    selected_set = set()
    
    for line in sorted_lines:
        if len(selected) == k:
            break
        # Check for duplicates (already handled by set, but canonical form ensures uniqueness)
        selected.append(line)
        selected_set.add(line)
        
    return np.array(selected, dtype=np.int64)

def _local_search_lines(points, lines, deadline, max_iter=5000):
    """
    Local search: try swapping lines to improve incidence count.
    """
    best_lines = lines.copy()
    best_count = _count_incidences(points, best_lines)
    
    n_lines = len(best_lines)
    
    # Precompute point coords
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    
    # Generate a pool of candidate lines for swaps
    # We can reuse the pool from greedy or generate new
    # For efficiency, let's generate a moderate pool
    candidate_pool = _generate_line_pool(points, max_lines=2000)
    
    # Filter out lines already in best_lines to save time? 
    # Or just keep them and check validity.
    
    # Precompute counts for candidate pool
    cand_counts = {}
    for line in candidate_pool:
        A, B, C = line
        vals = A * px + B * py + C
        cand_counts[line] = int(np.sum(vals == 0))
        
    # Sort candidates by count
    sorted_candidates = sorted(cand_counts.keys(), key=lambda l: cand_counts[l], reverse=True)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        # Try swapping each line in best_lines with a better candidate
        for i in range(n_lines):
            if time.time() > deadline:
                break
                
            current_line = tuple(best_lines[i])
            current_count = cand_counts.get(current_line, _count_incidences(points, np.array([current_line])))
            
            # Find a candidate with higher count
            for cand in sorted_candidates:
                if cand == current_line:
                    continue
                if cand in best_lines:
                    continue
                    
                cand_count = cand_counts[cand]
                
                # If candidate is better, swap
                if cand_count > current_count:
                    new_lines = best_lines.copy()
                    new_lines[i] = cand
                    new_count = best_count - current_count + cand_count
                    
                    if new_count > best_count:
                        best_count = new_count
                        best_lines = new_lines
                        improved = True
                        break
                        
        if not improved:
            # Try random swaps to escape local optima
            for _ in range(10):
                if time.time() > deadline:
                    break
                i = random.randint(0, n_lines - 1)
                cand = random.choice(sorted_candidates[:100]) # Top 100 candidates
                
                if cand in best_lines:
                    continue
                
                current_line = tuple(best_lines[i])
                current_count = cand_counts.get(current_line, _count_incidences(points, np.array([current_line])))
                cand_count = cand_counts[cand]
                
                new_lines = best_lines.copy()
                new_lines[i] = cand
                new_count = _count_incidences(points, new_lines)
                
                if new_count > best_count:
                    best_count = new_count
                    best_lines = new_lines
                    improved = True
                    break
                    
        if not improved:
            # Stagnation: try random restart of a few lines
            break
            
    return best_lines, best_count

def _solve_for_grid(rows, cols, deadline):
    """
    Solve for a specific grid configuration.
    """
    points = _generate_grid_points(rows, cols)
    if len(points) != 54:
        return None, None, 0
        
    pool = _generate_line_pool(points, max_lines=3000)
    
    # Greedy initialization
    lines = _greedy_select_lines(points, pool, 27)
    
    # Local search
    lines, count = _local_search_lines(points, lines, deadline)
    
    return points, lines, count

def szemeredi_trotter_incidences():
    """
    Main function to find optimal point-line incidences.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_lines = None
    best_count = 0
    
    # Try different grid configurations
    # 54 factors: 1x54, 2x27, 3x18, 6x9
    # Also try non-rectangular grids or shifted grids?
    # For now, stick to rectangular grids.
    
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    for r, c in grid_configs:
        if time.time() > deadline - 30:
            break
            
        p, l, count = _solve_for_grid(r, c, deadline)
        if p is not None and count > best_count:
            best_count = count
            best_points = p
            best_lines = l
            
    # If no valid solution found (shouldn't happen), return a default
    if best_points is None:
        best_points = _generate_grid_points(6, 9)
        best_lines = np.zeros((27, 3), dtype=np.int64)
        for i in range(27):
            best_lines[i] = [0, 1, -i] # y = i
            
    return best_points, best_lines

# EVOLVE-BLOCK-END
