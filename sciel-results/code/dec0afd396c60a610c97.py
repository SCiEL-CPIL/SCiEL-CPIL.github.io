# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
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
    Count total incidences between points and lines.
    points: (N, 2) array
    lines: (K, 3) array
    Returns: total incidences (int)
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    # Vectorized check
    # points: (N, 2), lines: (K, 3)
    # We can do this efficiently by iterating lines and checking all points
    # Or using broadcasting if small enough. N=54, K=27 is tiny.
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_line_pool(max_coord=10):
    """
    Generate a diverse pool of lines with small integer coefficients.
    Focus on lines that intersect within the expected point range.
    """
    pool = set()
    
    # 1. Grid lines: x=c, y=c
    for c in range(-max_coord, max_coord + 1):
        pool.add(_normalize_line(1, 0, -c))
        pool.add(_normalize_line(0, 1, -c))
        
    # 2. Diagonals: x+y=c, x-y=c
    for c in range(-2*max_coord, 2*max_coord + 1):
        pool.add(_normalize_line(1, 1, -c))
        pool.add(_normalize_line(1, -1, c))
        
    # 3. Other simple slopes: y = 2x + c, y = 0.5x + c (2y = x + c'), etc.
    # A*x + B*y + C = 0
    # Small A, B pairs
    slopes = [
        (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
        (1, -2), (2, -1), (1, -3), (3, -1), (2, -3), (3, -2)
    ]
    for A, B in slopes:
        # C range to keep intersections near origin
        c_range = 2 * max_coord
        for c in range(-c_range, c_range + 1):
            pool.add(_normalize_line(A, B, c))
            
    # 4. Random lines with small coeffs
    for _ in range(500):
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        if A != 0 or B != 0:
            pool.add(_normalize_line(A, B, C))
            
    return list(pool)

def _find_best_assignment(points, lines, max_time=10):
    """
    Given a set of candidate lines, find the best subset of 27 lines 
    and assign points to them to maximize incidences?
    Actually, the problem is:
    We have 54 points (fixed set of coordinates) and we must choose 27 lines.
    The score is sum of incidences.
    
    Alternative View:
    We choose 54 points and 27 lines.
    
    Let's stick to the "Fixed Line Pool" strategy:
    1. Generate a pool of ~200 lines.
    2. We need to pick 27 lines from this pool.
    3. We need to place 54 points.
    
    This is still hard. Let's invert:
    1. Start with a good grid of points (e.g. 9x6).
    2. Identify the "best" lines for this grid (top 27 by incidence).
    3. Now, treat the 27 lines as FIXED.
    4. Optimize the 54 points to maximize incidences with these 27 fixed lines.
    5. Then, re-evaluate the top 27 lines for the new points.
    6. Iterate.
    
    This "Alternating Minimization" is the key difference.
    """
    pass # Not used directly, logic is in main search

def _optimize_points_for_lines(points, lines, max_time=20):
    """
    Given fixed lines, optimize the positions of the 54 points to maximize incidences.
    We use a local search: move one point at a time to a nearby integer coordinate
    that increases the total incidence count.
    """
    start_time = time.time()
    deadline = start_time + max_time
    current_score = _count_incidences(points, lines)
    
    # Precompute line coefficients for speed
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    def score_point(pt):
        # Returns number of lines this point lies on
        x, y = pt
        vals = A * x + B * y + C
        return np.sum(vals == 0)
    
    # Local search
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 1 and no_improvement < max_no_improvement:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(len(points)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 1:
                break
                
            old_pt = points[idx]
            old_score = score_point(old_pt)
            
            # Generate candidates: 8-neighbors + some random nearby
            candidates = set()
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((old_pt[0] + dx, old_pt[1] + dy))
            
            # Add a few random jumps
            for _ in range(5):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                candidates.add((old_pt[0] + dx, old_pt[1] + dy))
            
            best_cand = None
            best_cand_score = old_score
            
            for c in candidates:
                # Check if c is already used by another point
                # This check is O(N). N=54. Acceptable.
                # We can optimize by maintaining a set of used points, 
                # but for now, simple check.
                is_used = False
                for j, p in enumerate(points):
                    if j != idx and p[0] == c[0] and p[1] == c[1]:
                        is_used = True
                        break
                if is_used:
                    continue
                
                new_score = score_point(c)
                if new_score > best_cand_score:
                    best_cand_score = new_score
                    best_cand = c
            
            if best_cand is not None:
                points[idx] = best_cand
                current_score += (best_cand_score - old_score)
                improved = True
                no_improvement = 0
            else:
                no_improvement += 1
                
        if not improved:
            break
            
    return points, current_score

def _optimize_lines_for_points(points, lines, max_time=20):
    """
    Given fixed points, optimize the selection of 27 lines from a pool
    to maximize incidences.
    We start with the current lines, and try to swap one line with another
    from the pool if it improves the score.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Precompute incidence of each point on each current line?
    # Actually, we just need to calculate the gain of swapping line i with line j.
    # Gain = (incidences of j - incidences of i) - (overlap correction?)
    # No, lines are distinct. Incidences are additive.
    # Score = sum(count(line_k) for k in 1..27)
    # If we swap line i with line j:
    # New Score = Score - count(line_i) + count(line_j)
    # So we just need count(line_j) for all j in pool.
    
    # Calculate count for all lines in the current set
    current_counts = np.zeros(len(lines), dtype=int)
    for i, line in enumerate(lines):
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        current_counts[i] = np.sum(vals == 0)
    
    current_score = np.sum(current_counts)
    
    # We need a pool of candidate lines to swap with.
    # Let's generate a pool based on the current points' bounding box.
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    pool = _generate_line_pool(max_coord=max(max_x, max_y, 10) + 5)
    
    # Precompute counts for all lines in the pool
    # This might be expensive if pool is large (2000 lines * 54 points).
    # 108k ops. Very fast.
    pool_counts = {}
    for line in pool:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        pool_counts[line] = np.sum(vals == 0)
        
    # Sort pool by count descending to prioritize good swaps
    sorted_pool = sorted(pool_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Local search: try to swap each current line with a better one from the pool
    no_improvement = 0
    max_no_improvement = 20
    
    while time.time() < deadline - 1 and no_improvement < max_no_improvement:
        improved = False
        indices = list(range(len(lines)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 1:
                break
                
            current_line = tuple(lines[i])
            current_count = current_counts[i]
            
            # Find the best line in the pool that is NOT currently in our set
            # and has a higher count
            best_swap_line = None
            best_swap_count = current_count
            
            # Iterate through sorted pool
            for line, count in sorted_pool:
                if count <= best_swap_count:
                    break # Since sorted, no more better lines
                
                # Check if this line is already in our current set
                if line == current_line:
                    continue
                
                is_in_set = False
                for j, l in enumerate(lines):
                    if j != i and tuple(l) == line:
                        is_in_set = True
                        break
                
                if not is_in_set:
                    best_swap_line = line
                    best_swap_count = count
                    break # Take the first (best) one found
                    
            if best_swap_line is not None:
                lines[i] = best_swap_line
                current_counts[i] = best_swap_count
                current_score += (best_swap_count - current_count)
                improved = True
                no_improvement = 0
            else:
                no_improvement += 1
                
        if not improved:
            break
            
    return lines, current_score

def _constructive_search(max_time=580):
    """
    Strategy:
    1. Initialize with a 9x6 grid.
    2. Alternating Minimization:
       a. Fix lines, optimize points (local search).
       b. Fix points, optimize lines (greedy swap from pool).
    3. Repeat until time runs out.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial seed: 9x6 grid
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Get initial top lines
    lines = np.array(_get_top_lines(points, 27), dtype=np.int64)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Alternating optimization loop
    while time.time() < deadline - 20:
        # Step 1: Optimize points for fixed lines
        # Allocate half the remaining time? Or fixed chunks?
        # Let's use 10s for points, 10s for lines.
        time_left = deadline - time.time()
        if time_left < 30:
            break
            
        chunk_time = min(10, time_left / 2 - 5)
        
        points, score_p = _optimize_points_for_lines(points, lines, max_time=chunk_time)
        
        # Step 2: Optimize lines for fixed points
        lines, score_l = _optimize_lines_for_points(points, lines, max_time=chunk_time)
        
        # Update best
        current_score = _count_incidences(points, lines)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            best_lines = lines.copy()
            
    return best_points, best_lines, best_score

def _get_top_lines(points, k=27):
    """
    Find the k lines with the most incidences among all lines defined by pairs of points.
    """
    n = len(points)
    pair_counts = {}
    
    # Iterate over all pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                pair_counts[key] = pair_counts.get(key, 0) + 1
                
    # Convert pair counts to point counts
    line_point_counts = {}
    for key, pc in pair_counts.items():
        disc = 1 + 8 * pc
        sqrt_disc = int(np.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k_pts = (1 + sqrt_disc) // 2
            if k_pts * (k_pts - 1) // 2 == pc:
                line_point_counts[key] = k_pts
            else:
                line_point_counts[key] = 2
        else:
            line_point_counts[key] = 2

    # Sort by count descending
    sorted_lines = sorted(line_point_counts.items(), key=lambda item: item[1], reverse=True)
    
    top_keys = [key for key, _ in sorted_lines[:k]]
    
    # Pad if necessary
    if len(top_keys) < k:
        existing = set(top_keys)
        for p in points:
            x, y = p
            candidates = [
                _normalize_line(1, 0, -x),
                _normalize_line(0, 1, -y),
                _normalize_line(1, 1, -(x+y)),
                _normalize_line(1, -1, x-y)
            ]
            for c in candidates:
                if c not in existing and c != (0,0,0):
                    top_keys.append(c)
                    existing.add(c)
                    if len(top_keys) == k:
                        break
            if len(top_keys) == k:
                break
        while len(top_keys) < k:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            C = random.randint(-100, 100)
            key = _normalize_line(A, B, C)
            if key not in existing and key != (0,0,0):
                top_keys.append(key)
                existing.add(key)

    return top_keys

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses alternating minimization of points and lines.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    try:
        best_points, best_lines, best_score = _constructive_search(max_time=timeout - 10)
    except Exception:
        # Fallback to simple grid if something goes wrong
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = np.array(_get_top_lines(best_points, 27), dtype=np.int64)
        
    # Final validation and formatting
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for invalid lines
    for l in lines_out:
        if l[0] == 0 and l[1] == 0:
            xs = np.arange(1, 10)
            ys = np.arange(1, 7)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
            break
            
    # Check for duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    if len(lines_set) < 27:
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)

    return points_out, lines_out

# EVOLVE-BLOCK-END
