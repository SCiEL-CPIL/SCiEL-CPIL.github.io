# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    x = points[:, 0][:, np.newaxis]
    y = points[:, 1][:, np.newaxis]
    A = lines[:, 0][np.newaxis, :]
    B = lines[:, 1][np.newaxis, :]
    C = lines[:, 2][np.newaxis, :]
    vals = x * A + y * B + C
    return int(np.sum(vals == 0))

def _get_best_lines(points, M, current_lines=None):
    """
    Find the M distinct lines with the highest incidence counts for the given points.
    Returns (lines_array, incidence_count)
    """
    N = len(points)
    if N < 2:
        # Degenerate case
        lines = np.zeros((M, 3), dtype=np.int64)
        lines[:, 0] = 1
        lines[:, 2] = np.arange(M)
        return lines, 0

    # Generate all unique lines defined by pairs of points
    line_map = {} # norm -> (A, B, C, count)
    
    # Optimization: Instead of checking all N^2 pairs, we can be smarter, 
    # but for N=54, N^2/2 = 1431 pairs, which is very fast.
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            norm = _normalize_line(A, B, C)
            
            if norm in line_map:
                # Already seen this line, increment count
                A_old, B_old, C_old, count_old = line_map[norm]
                line_map[norm] = (A_old, B_old, C_old, count_old + 1)
            else:
                # First time seeing this line, count points on it
                # We know i and j are on it. Check others? 
                # For efficiency, we can just count all points for this line now.
                # Since N is small, this is O(N) per unique line.
                # Max unique lines is ~1431. 1431 * 54 is ~77k ops. Very fast.
                count = 0
                for k in range(N):
                    if A * points[k, 0] + B * points[k, 1] + C == 0:
                        count += 1
                line_map[norm] = (A, B, C, count)

    # Sort lines by count descending
    sorted_lines = sorted(line_map.values(), key=lambda x: x[3], reverse=True)
    
    # Select top M distinct lines
    selected = []
    for line in sorted_lines:
        if len(selected) == M:
            break
        selected.append(line)
        
    # If we don't have enough lines (unlikely for N=54, M=27), pad with arbitrary distinct lines
    if len(selected) < M:
        # Generate random lines
        existing_norms = set(_normalize_line(l[0], l[1], l[2]) for l in selected)
        while len(selected) < M:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            C = random.randint(-20, 20)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in existing_norms:
                selected.append((A, B, C, 0))
                existing_norms.add(norm)
            else:
                C += 1
                norm = _normalize_line(A, B, C)
                if norm not in existing_norms:
                    selected.append((A, B, C, 0))
                    existing_norms.add(norm)

    lines_arr = np.array([(l[0], l[1], l[2]) for l in selected], dtype=np.int64)
    total_inc = sum(l[3] for l in selected)
    
    return lines_arr, total_inc

def _generate_initial_config():
    """
    Generate a random configuration where points are clustered on a few lines.
    This helps the greedy line selector find high-incidence lines immediately.
    """
    N = 54
    M = 27
    
    # Choose 3 to 5 base lines
    num_base = random.randint(3, 5)
    base_lines = []
    base_norms = set()
    
    for _ in range(num_base):
        while True:
            # Small integer coefficients for dense lattice points
            A = random.randint(-3, 3)
            B = random.randint(-3, 3)
            if A == 0 and B == 0:
                continue
            C = random.randint(-10, 10)
            norm = _normalize_line(A, B, C)
            if norm not in base_norms:
                base_norms.add(norm)
                base_lines.append((A, B, C))
                break
    
    points_list = []
    used_points = set()
    
    # Distribute points roughly evenly among base lines
    points_per_line = [N // num_base] * num_base
    for i in range(N % num_base):
        points_per_line[i] += 1
        
    for idx, (A, B, C) in enumerate(base_lines):
        count_needed = points_per_line[idx]
        
        # Find integer points on this line
        # A*x + B*y + C = 0
        candidates = []
        # Search a reasonable box
        for x in range(-15, 15):
            if A == 0:
                if B != 0 and (-C) % B == 0:
                    y = -C // B
                    candidates.append((x, y))
            else:
                val = -A * x - C
                if B == 0:
                    if val == 0:
                        # Infinite solutions for y, pick one
                        candidates.append((x, 0))
                else:
                    if val % B == 0:
                        y = val // B
                        candidates.append((x, y))
        
        random.shuffle(candidates)
        
        for p in candidates:
            if len(points_list) >= N:
                break
            if p not in used_points:
                points_list.append(p)
                used_points.add(p)
                
    # If we didn't get enough points (e.g., line was sparse or range too small), add random points
    while len(points_list) < N:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in used_points:
            points_list.append((x, y))
            used_points.add((x, y))
            
    points = np.array(points_list[:N], dtype=np.int64)
    
    # Get best lines for this point set
    lines, inc = _get_best_lines(points, M)
    
    return points, lines, inc

def _refine_lines(points, lines, M):
    """
    Check if current lines are the best possible for the given points.
    If not, update lines to the best set.
    Returns new lines and new incidence count.
    """
    new_lines, new_inc = _get_best_lines(points, M)
    return new_lines, new_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin
    
    N = 54
    M = 27
    
    best_points = None
    best_lines = None
    best_I = -1
    
    # Run multiple restarts with different random seeds
    restarts = 10
    time_per_restart = (deadline - start_time) / restarts
    
    for r in range(restarts):
        restart_start = time.time()
        restart_deadline = min(start_time + time_per_restart * (r + 1), deadline - 10.0)
        
        if time.time() > restart_deadline:
            break
            
        # 1. Initialize
        points, lines, I = _generate_initial_config()
        
        if I > best_I:
            best_I = I
            best_points = points.copy()
            best_lines = lines.copy()
            
        # 2. Simulated Annealing / Local Search
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9999
        iterations = 0
        no_improve_count = 0
        
        # Precompute point set for duplicate check
        point_set = set(map(tuple, points.tolist()))
        
        while time.time() < restart_deadline:
            iterations += 1
            
            # Move a point
            i = random.randint(0, N - 1)
            old_px, old_py = int(points[i, 0]), int(points[i, 1])
            
            # Calculate current incidence for this point
            old_row = (lines[:, 0] * old_px + lines[:, 1] * old_py + lines[:, 2]) == 0
            old_row_sum = int(np.sum(old_row))
            
            best_delta = -1000
            best_new_px, best_new_py = None, None
            best_new_row = None
            
            # Try neighbors in a small spiral
            # To speed up, we can limit the search range
            max_d = 3
            for dx in range(-max_d, max_d + 1):
                for dy in range(-max_d, max_d + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_px = old_px + dx
                    new_py = old_py + dy
                    
                    if (new_px, new_py) in point_set:
                        continue
                    
                    new_row = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2]) == 0
                    new_row_sum = int(np.sum(new_row))
                    delta = new_row_sum - old_row_sum
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_px = new_px
                        best_new_py = new_py
                        best_new_row = new_row
            
            if best_new_px is None:
                continue
                
            # Acceptance criterion
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Update state
                point_set.discard((old_px, old_py))
                point_set.add((best_new_px, best_new_py))
                points[i, 0] = best_new_px
                points[i, 1] = best_new_py
                I += best_delta
                
                # Periodic Line Refinement
                # Every 20 iterations, check if we should update the lines
                if iterations % 20 == 0:
                    new_lines, new_I = _refine_lines(points, M)
                    delta_lines = new_I - I
                    # Always accept line improvement if it's significant, or with some probability
                    if delta_lines > 0 or random.random() < np.exp(delta_lines / T):
                        lines = new_lines
                        I = new_I
                        
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Early termination for this restart if stuck
                if no_improve_count > 1000:
                    break
                    
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Final validation and cleanup
    # Ensure no duplicate points
    # Ensure no duplicate lines
    # This is usually handled by the logic above, but double check.
    
    # Return the best found
    return best_points, best_lines

# EVOLVE-BLOCK-END
