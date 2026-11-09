import numpy as np
import time
import random
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences exactly."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64).T
    B = lines[:, 1:2].astype(np.int64).T
    C = lines[:, 2:3].astype(np.int64).T
    total = px * A + py * B + C
    return int(np.sum(total == 0))

def _get_incidence_matrix(points, lines):
    """
    Returns a (N_pts, N_lines) boolean matrix where M[i, j] is True if point i is on line j.
    This allows fast delta calculation.
    """
    if len(points) == 0 or len(lines) == 0:
        return np.zeros((len(points), len(lines)), dtype=bool)
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64).T
    B = lines[:, 1:2].astype(np.int64).T
    C = lines[:, 2:3].astype(np.int64).T
    # M[i, j] corresponds to point i and line j
    # px is (N, 1), A is (1, M) -> result (N, M)
    total = px * A + py * B + C
    return (total == 0)

def _generate_initial_config(seed=42):
    """
    Generate an initial configuration of 54 points and 27 lines.
    We use a 6x9 grid as a strong baseline, then randomly perturb it slightly to avoid exact symmetry traps if needed, 
    but starting from a grid is usually best.
    """
    rng = random.Random(seed)
    
    # Start with 6x9 grid
    w, h = 6, 9
    xs = np.arange(1, w + 1)
    ys = np.arange(1, h + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate candidate lines
    candidates = set()
    max_slope = 4
    
    # Horizontal
    for c in range(1, h + 1):
        candidates.add(_normalize_line(0, 1, -c))
        
    # Vertical
    for c in range(1, w + 1):
        candidates.add(_normalize_line(1, 0, -c))
        
    # Slopes
    for A in range(-max_slope, max_slope + 1):
        for B in range(-max_slope, max_slope + 1):
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), abs(B))
            if g > 0:
                A_norm, B_norm = A // g, B // g
            else:
                A_norm, B_norm = A, B
            if A_norm < 0 or (A_norm == 0 and B_norm < 0):
                A_norm, B_norm = -A_norm, -B_norm
            
            # Determine range of C
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            vals = [A*1 + B*1, A*1 + B*h, A*w + B*1, A*w + B*h]
            min_sum = min(vals)
            max_sum = max(vals)
            
            for C in range(-max_sum, -max_sum + 1):
                candidates.add(_normalize_line(A_norm, B_norm, C))
                
    # Convert to list
    candidates = list(candidates)
    
    # Select top 27 lines based on initial incidence
    cand_arr = np.array(candidates, dtype=np.int64)
    inc_counts = _compute_all_incidences(points, cand_arr)
    top_indices = np.argsort(inc_counts)[::-1][:27]
    initial_lines = np.array([candidates[i] for i in top_indices], dtype=np.int64)
    
    return points, initial_lines, candidates

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

def _joint_simulated_annealing(points, lines, candidates, time_limit, start_time):
    """
    Joint Simulated Annealing on Points and Lines.
    State: (points, lines)
    Move: 
      1. Move a point to a nearby integer coordinate (within radius R).
      2. Swap a line with a random candidate line.
    Score: Total incidences.
    """
    n_pts = len(points)
    n_lines = len(lines)
    
    current_points = points.copy()
    current_lines = lines.copy()
    
    # Precompute incidence matrix for current state
    current_inc_matrix = _get_incidence_matrix(current_points, current_lines)
    current_score = int(np.sum(current_inc_matrix))
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Candidate lines array for quick lookup
    cand_arr = np.array(candidates, dtype=np.int64)
    cand_inc_counts = _compute_all_incidences(current_points, cand_arr)
    
    # Map line tuple to index for O(1) lookup if needed, but we swap by index
    # We need to know if a candidate line is already in current_lines to avoid duplicates?
    # The problem says "two rows describing the same line" is invalid.
    # So we must ensure current_lines has unique normalized lines.
    
    T = 50.0
    cooling_rate = 0.999
    rng = random.Random(42)
    
    # Radius for point moves
    point_radius = 2
    
    while time.time() < start_time + time_limit:
        if T < 0.1:
            break
            
        # Choose move type: 50% point, 50% line
        if rng.random() < 0.5:
            # --- Point Move ---
            i = rng.randrange(n_pts)
            old_x, old_y = current_points[i]
            
            # Pick a random neighbor
            dx = rng.randint(-point_radius, point_radius)
            dy = rng.randint(-point_radius, point_radius)
            if dx == 0 and dy == 0:
                continue
                
            new_x, new_y = old_x + dx, old_y + dy
            
            # Validation: positive coordinates (task implies integer coords, usually positive in these grids, 
            # but technically any integer. However, large negatives might cause issues with line generation range?
            # The task says "integer (x,y)". It doesn't restrict to positive. 
            # But our candidate lines are generated for x,y in [1, W]x[1, H].
            # If we move points far away, they won't be on any candidate lines.
            # So we should constrain moves to stay near the original grid area or expand candidate set.
            # For efficiency, let's keep points within [1, 10] x [1, 10] roughly, 
            # or just accept that moving far out reduces score and SA will reject.
            
            # Check duplicate points
            is_dup = False
            for j in range(n_pts):
                if i == j:
                    continue
                if current_points[j][0] == new_x and current_points[j][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Calculate delta in score
            # Old incidences for point i: current_inc_matrix[i, :]
            # New incidences for point i: Check which lines pass through (new_x, new_y)
            # Line j: A_j*new_x + B_j*new_y + C_j == 0
            
            # We can compute the new row of the incidence matrix for point i
            A = current_lines[:, 0]
            B = current_lines[:, 1]
            C = current_lines[:, 2]
            new_row = (A * new_x + B * new_y + C == 0)
            
            old_row = current_inc_matrix[i, :]
            
            delta = np.sum(new_row) - np.sum(old_row)
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                # Accept
                current_points[i] = [new_x, new_y]
                current_inc_matrix[i, :] = new_row
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
                    
                    # Update candidate incidence counts? 
                    # Moving a point changes the incidence count for ALL candidate lines.
                    # This is O(N_cands * N_lines) if done naively, or O(N_cands) if we just check the moved point.
                    # cand_inc_counts[j] is sum over points.
                    # delta for cand j = (1 if point i on cand j else 0) - (1 if point i was on cand j else 0)
                    # We can update cand_inc_counts efficiently.
                    # But we don't have the incidence matrix for ALL candidates, only counts.
                    # To update counts, we need to know if point i is on candidate j.
                    # This requires checking A_j*new_x + B_j*new_y + C_j == 0 for all j.
                    # O(N_cands). N_cands is ~100-200. This is fast.
                    
                    # Let's update cand_inc_counts
                    old_row_cand = (cand_arr[:, 0] * old_x + cand_arr[:, 1] * old_y + cand_arr[:, 2] == 0)
                    new_row_cand = (cand_arr[:, 0] * new_x + cand_arr[:, 1] * new_y + cand_arr[:, 2] == 0)
                    cand_inc_counts += (new_row_cand.astype(np.int64) - old_row_cand.astype(np.int64))
                    
        else:
            # --- Line Swap ---
            i = rng.randrange(n_lines)
            old_line_idx_in_cand = None
            # Find index of current_lines[i] in candidates
            # This is O(N_cands) if linear search. N_cands is small.
            old_line_tuple = tuple(current_lines[i])
            for k, c in enumerate(candidates):
                if c == old_line_tuple:
                    old_line_idx_in_cand = k
                    break
            if old_line_idx_in_cand is None:
                # Should not happen if we only swap with candidates
                continue
                
            # Pick a new candidate line
            # Weighted by incidence count? Or random?
            # Random is better for exploration.
            new_line_idx = rng.randrange(len(candidates))
            
            if new_line_idx == old_line_idx_in_cand:
                continue
                
            # Check if new line is already in current_lines
            new_line_tuple = candidates[new_line_idx]
            is_dup_line = False
            for k in range(n_lines):
                if k == i:
                    continue
                if tuple(current_lines[k]) == new_line_tuple:
                    is_dup_line = True
                    break
            if is_dup_line:
                continue
                
            # Calculate delta
            # Old line contribution: sum over points of (point on old_line)
            # This is sum(current_inc_matrix[:, i])
            old_line_score = int(np.sum(current_inc_matrix[:, i]))
            
            # New line contribution: sum over points of (point on new_line)
            # We have cand_inc_counts[new_line_idx] which is the total incidence of that candidate line against current_points.
            new_line_score = int(cand_inc_counts[new_line_idx])
            
            delta = new_line_score - old_line_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                # Accept
                current_lines[i] = new_line_tuple
                current_score += delta
                
                # Update incidence matrix column i
                new_line_arr = np.array(new_line_tuple)
                A = new_line_arr[0]
                B = new_line_arr[1]
                C = new_line_arr[2]
                px = current_points[:, 0]
                py = current_points[:, 1]
                current_inc_matrix[:, i] = (A * px + B * py + C == 0)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
                    
        T *= cooling_rate
        
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    # 1. Generate initial configuration
    points, lines, candidates = _generate_initial_config(seed=42)
    
    # 2. Run Joint Simulated Annealing
    # Allocate most time to SA
    sa_time = 500
    sa_points, sa_lines, sa_score = _joint_simulated_annealing(points, lines, candidates, sa_time, start_time)
    
    # 3. Final Local Search Refinement (Greedy)
    # Try to improve by swapping lines with higher incidence candidates
    # And moving points to nearby spots that increase incidence
    
    remaining_time = deadline - time.time() - 10
    if remaining_time > 5:
        # Simple greedy line swap
        cand_arr = np.array(candidates, dtype=np.int64)
        inc_counts = _compute_all_incidences(sa_points, cand_arr)
        
        # Sort candidates by incidence
        sorted_cands = np.argsort(inc_counts)[::-1]
        
        # Try to replace each line with a better one from the top list
        # We must ensure no duplicates
        current_lines_set = {tuple(l) for l in sa_lines}
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_gain = 0
            best_swap = None
            
            for i in range(len(sa_lines)):
                old_line = tuple(sa_lines[i])
                old_score = int(np.sum((sa_lines[i][0] * sa_points[:, 0] + sa_lines[i][1] * sa_points[:, 1] + sa_lines[i][2]) == 0))
                
                # Look for a replacement in top 50 candidates
                for idx in sorted_cands[:50]:
                    new_line = candidates[idx]
                    if new_line in current_lines_set:
                        continue
                    new_score = int(inc_counts[idx])
                    gain = new_score - old_score
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, new_line)
                        
            if best_swap:
                i, new_line = best_swap
                old_line = tuple(sa_lines[i])
                current_lines_set.remove(old_line)
                current_lines_set.add(new_line)
                sa_lines[i] = new_line
                improved = True
                # Recalculate inc_counts? 
                # Line swap doesn't affect other lines' incidence counts, only the score of the set.
                # But inc_counts is per candidate line, independent of other lines. So it remains valid.
                
        # Greedy point move
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_gain = 0
            best_move = None
            points_set = {tuple(p) for p in sa_points}
            
            for i in range(len(sa_points)):
                x, y = sa_points[i]
                old_score = int(np.sum((sa_lines[:, 0] * x + sa_lines[:, 1] * y + sa_lines[:, 2]) == 0))
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in points_set:
                            continue
                        new_score = int(np.sum((sa_lines[:, 0] * nx + sa_lines[:, 1] * ny + sa_lines[:, 2]) == 0))
                        gain = new_score - old_score
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, nx, ny)
                            
            if best_move:
                i, nx, ny = best_move
                old_pt = tuple(sa_points[i])
                points_set.remove(old_pt)
                points_set.add((nx, ny))
                sa_points[i] = [nx, ny]
                improved = True
                
    # Final validation
    # Ensure no duplicate points
    # Ensure no duplicate lines
    # Ensure shapes are correct
    
    # Check duplicate points
    seen = set()
    valid_points = []
    for p in sa_points:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            valid_points.append(p)
    if len(valid_points) < 54:
        # Fallback to grid
        w, h = 6, 9
        xs = np.arange(1, w + 1)
        ys = np.arange(1, h + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        valid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Regenerate lines for this grid
        # ... use initial lines logic or just return grid and best lines found
        
    sa_points = np.array(valid_points, dtype=np.int64)
    
    # Check duplicate lines
    seen_lines = set()
    valid_lines = []
    for l in sa_lines:
        t = tuple(l)
        if t not in seen_lines:
            seen_lines.add(t)
            valid_lines.append(l)
    if len(valid_lines) < 27:
        # Pad with random valid lines? Or just use initial grid lines
        # For now, if we lost lines, we might be invalid. 
        # Let's ensure we have 27 unique lines.
        # If not, we can add horizontal/vertical lines that are not present.
        for c in range(1, 20):
            line = _normalize_line(0, 1, -c)
            if line not in seen_lines:
                seen_lines.add(line)
                valid_lines.append(line)
            if len(valid_lines) == 27:
                break
            line2 = _normalize_line(1, 0, -c)
            if line2 not in seen_lines:
                seen_lines.add(line2)
                valid_lines.append(line2)
            if len(valid_lines) == 27:
                break
                
    sa_lines = np.array(valid_lines[:27], dtype=np.int64)
    
    return sa_points, sa_lines
