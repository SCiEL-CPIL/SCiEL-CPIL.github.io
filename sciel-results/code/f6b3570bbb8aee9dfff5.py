import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Handle zero C case for gcd
    abs_g = gcd(abs(A), abs(B))
    if C != 0:
        abs_g = gcd(abs_g, abs(C))
    if abs_g == 0:
        abs_g = 1
        
    A //= abs_g
    B //= abs_g
    C //= abs_g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def structured_search(candidate_points, n_lines=27, time_limit=60):
    """
    Construct a 9x9 grid from 9 vertical and 9 horizontal lines, then add 9 diagonal lines
    (4 slope +1, 5 slope -1). Select 54 points from the 81 grid points to maximize incidences.
    """
    start_time = time.time()
    
    best_I = -1
    best_lines = None
    best_points = None
    
    # Try different grid configurations
    # Grid: x values from x0 to x0+8, y values from y0 to y0+8
    # Diagonals: x-y=c and x+y=c for various c
    
    for x0 in range(-10, 11, 5):  # Try different grid positions
        for y0 in range(-10, 11, 5):
            if time.time() - start_time > time_limit:
                break
            
            # 9 vertical lines: x = x0, x0+1, ..., x0+8
            v_lines = [(1, 0, -(x0+i)) for i in range(9)]
            # 9 horizontal lines: y = y0, y0+1, ..., y0+8
            h_lines = [(0, 1, -(y0+i)) for i in range(9)]
            
            # The 81 grid points
            grid_pts = [(x0+i, y0+j) for i in range(9) for j in range(9)]
            
            # Try different diagonal configurations
            # Diagonals x-y=c: c ranges from (x0)-(y0+8) to (x0+8)-(y0)
            # Diagonals x+y=c: c ranges from (x0)+(y0) to (x0+8)+(y0+8)
            
            d_plus_options = [(1, -1, -(x0+i-y0-j)) for i in range(9) for j in range(9)]
            d_minus_options = [(1, 1, -(x0+i+y0+j)) for i in range(9) for j in range(9)]
            
            # Unique diagonals
            d_plus_unique = list(set(d_plus_options))
            d_minus_unique = list(set(d_minus_options))
            
            # For each diagonal, count how many grid points it passes through
            d_plus_counts = {}
            for d in d_plus_unique:
                A, B, C = d
                cnt = sum(1 for (px, py) in grid_pts if A*px + B*py + C == 0)
                d_plus_counts[d] = cnt
            
            d_minus_counts = {}
            for d in d_minus_unique:
                A, B, C = d
                cnt = sum(1 for (px, py) in grid_pts if A*px + B*py + C == 0)
                d_minus_counts[d] = cnt
            
            # Sort by count descending
            d_plus_sorted = sorted(d_plus_unique, key=lambda d: d_plus_counts[d], reverse=True)
            d_minus_sorted = sorted(d_minus_unique, key=lambda d: d_minus_counts[d], reverse=True)
            
            # Try different splits: 4 D+, 5 D- or 5 D+, 4 D-
            for n_plus in [4, 5]:
                n_minus = 9 - n_plus
                
                # Take top n_plus D+ and top n_minus D-
                sel_plus = d_plus_sorted[:n_plus]
                sel_minus = d_minus_sorted[:n_minus]
                
                all_lines = v_lines + h_lines + sel_plus + sel_minus
                all_lines = [normalize_line(A, B, C) for A, B, C in all_lines]
                all_lines = [l for l in all_lines if l is not None]
                
                # Check uniqueness
                seen = set()
                unique_lines = []
                for l in all_lines:
                    if l not in seen:
                        seen.add(l)
                        unique_lines.append(l)
                
                if len(unique_lines) < 27:
                    continue
                
                lines_arr = np.array(unique_lines[:27], dtype=np.int64)
                
                # For each grid point, count how many lines it's on
                pt_counts = []
                for (px, py) in grid_pts:
                    cnt = 0
                    for (A, B, C) in unique_lines[:27]:
                        if A*px + B*py + C == 0:
                            cnt += 1
                    pt_counts.append(cnt)
                
                # Select top 54 points
                pt_counts_arr = np.array(pt_counts)
                top_indices = np.argsort(pt_counts_arr)[::-1][:54]
                selected_pts = [grid_pts[i] for i in top_indices]
                
                points_arr = np.array(selected_pts, dtype=np.int64)
                I = count_incidences(points_arr, lines_arr)
                
                if I > best_I:
                    best_I = I
                    best_lines = lines_arr.copy()
                    best_points = points_arr.copy()
    
    # Also try the original approach as fallback
    if time.time() - start_time < time_limit - 10:
        v_intercepts = list(range(-40, 41))
        h_intercepts = list(range(-40, 41))
        d_plus_intercepts = list(range(-80, 81))
        d_minus_intercepts = list(range(-80, 81))
        
        candidates = []
        for c in v_intercepts:
            candidates.append((1, 0, -c, 'V'))
        for c in h_intercepts:
            candidates.append((0, 1, -c, 'H'))
        for c in d_plus_intercepts:
            candidates.append((1, -1, -c, 'D+'))
        for c in d_minus_intercepts:
            candidates.append((1, 1, -c, 'D-'))
            
        P = candidate_points.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        
        line_counts = np.zeros(len(candidates), dtype=int)
        for i, (A, B, C, fam) in enumerate(candidates):
            vals = A * px + B * py + C
            line_counts[i] = np.sum(vals == 0)
            
        sorted_indices = np.argsort(line_counts)[::-1]
        
        variations = [
            {'V': 9, 'H': 9, 'D+': 5, 'D-': 4},
            {'V': 8, 'H': 8, 'D+': 6, 'D-': 5},
            {'V': 9, 'H': 9, 'D+': 4, 'D-': 5},
        ]
        
        for var in variations:
            if time.time() - start_time > time_limit - 5:
                break
                
            rem = var.copy()
            sel_idx = []
            
            for idx in sorted_indices:
                if len(sel_idx) == 27:
                    break
                fam = candidates[idx][3]
                if rem.get(fam, 0) > 0:
                    sel_idx.append(idx)
                    rem[fam] -= 1
            
            if len(sel_idx) < 27:
                for idx in sorted_indices:
                    if idx not in sel_idx:
                        sel_idx.append(idx)
                    if len(sel_idx) == 27:
                        break
                        
            lines = np.array([candidates[i][:3] for i in sel_idx], dtype=np.int64)
            
            unique_lines = []
            seen = set()
            for l in lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and nl not in seen:
                    seen.add(nl)
                    unique_lines.append(nl)
                    
            if len(unique_lines) < 27:
                continue
                
            lines = np.array(unique_lines[:27], dtype=np.int64)
            points, _ = get_best_points_for_lines(lines, candidate_points)
            I = count_incidences(points, lines)
            
            if I > best_I:
                best_I = I
                best_lines = lines
                best_points = points
    
    return best_lines, best_points, best_I

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k indices
    top_indices = np.argsort(counts)[::-1][:k]
    best_points = candidate_points[top_indices]
    
    return best_points, int(counts[top_indices].max())

def generate_candidate_points(size=50, extra=2000):
    """Generate a grid of candidate points plus some random perturbations."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Add some random points in a slightly larger range for diversity
    rng = np.random.RandomState(123)
    rand_pts = rng.randint(-size-5, size+6, size=(extra, 2))
    
    all_pts = np.vstack([grid_pts, rand_pts])
    # Remove duplicates
    all_pts, _ = np.unique(all_pts, axis=0, return_index=True)
    return all_pts

def greedy_select_lines_from_points(points, n_lines=27, candidate_pool=None):
    """
    Greedily select lines that maximize total incidences with the given points.
    Uses a smarter approach: prioritize lines through 3+ points.
    """
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    n_pts = len(points)
    candidate_lines = []
    seen_lines = set()
    
    # Generate lines from pairs
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = dy
            B = -dx
            C = dx * points[i, 1] - dy * points[i, 0]
            nl = normalize_line(int(A), int(B), int(C))
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    # Pre-compute incidence counts for all candidate lines
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        cnt = int(np.sum(vals == 0))
        line_counts[nl] = cnt
    
    # Sort candidates by count descending for faster greedy
    sorted_candidates = sorted(candidate_lines, key=lambda l: line_counts[l], reverse=True)
    
    # Greedy selection with marginal gain consideration
    selected_lines = []
    selected_set = set()
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in sorted_candidates:
            if nl in selected_set:
                continue
            score = line_counts[nl]
            if score > best_score:
                best_score = score
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
    
    # Fill if needed with random lines
    rng = random.Random(42)
    while len(selected_lines) < n_lines:
        A = rng.randint(-7, 7)
        B = rng.randint(-7, 7)
        C = rng.randint(-35, 35)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_initial_lines(config_id):
    """Generate diverse initial line configurations."""
    lines = []
    if config_id == 0:
        # Grid + diagonals
        for i in range(9):
            lines.append((1, 0, -i))
        for j in range(6):
            lines.append((0, 1, -j))
        for c in range(-8, 4):
            lines.append((1, -1, -c))
    elif config_id == 1:
        # Various slopes
        a_vals = [0, 1, -1, 2, -2]
        b_vals = [0, 1, -1, 2, -2, 3, -3]
        for a in a_vals:
            for b in b_vals:
                lines.append((a, -1, b))
                if len(lines) >= 27:
                    break
            if len(lines) >= 27:
                break
        while len(lines) < 27:
            lines.append((1, 0, len(lines)))
    elif config_id == 2:
        # Pencil through origin
        pairs = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (3,1), (1,3), (3,2), (2,3), (3,4), (4,3), (5,1), (1,5)]
        for A, B in pairs:
            lines.append((A, -B, 0))
        for i in range(27 - len(lines)):
            lines.append((1, 1, i+1))
    elif config_id == 3:
        # Random
        random.seed(42)
        seen = set()
        while len(lines) < 27:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-10, 10)
            if A == 0 and B == 0:
                continue
            nl = normalize_line(A, B, C)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
                
    return np.array(lines[:27], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42, structured_pool=None):
    """
    Joint optimization of points and lines using simulated annealing.
    Improved: better move generation, faster evaluation, smarter restarts.
    Uses structured_pool to bias line replacements towards high-yield families.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 15.0
    cooling = 0.999
    min_T = 0.3
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Precompute a set of current line tuples for fast duplicate check
    current_line_set = set(tuple(l) for l in current_lines)
    
    while time.time() - start_time < time_limit:
        if no_improve > 300:
            T = max(T, 15.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.30:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if new_line_tup in current_line_set:
                continue
                
            current_lines[idx] = new_line
            current_line_set.discard(tuple(old_line))
            current_line_set.add(new_line_tup)
            
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                current_line_set.discard(new_line_tup)
                current_line_set.add(tuple(old_line))
                no_improve += 1
                
        elif move_type < 0.50:
            # Replace a line with a random line from the structured pool (V, H, D+, D-)
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            if structured_pool and len(structured_pool) > 0:
                # Pick a random line from the structured pool
                new_line = structured_pool[random.randint(0, len(structured_pool)-1)].copy()
            else:
                # Fallback to random small coefficients
                A = random.randint(-8, 8)
                B = random.randint(-8, 8)
                C = random.randint(-40, 40)
                new_line = normalize_line(A, B, C)
                if new_line is None:
                    continue
                new_line = np.array(new_line, dtype=np.int64)
            
            new_line_tup = tuple(new_line)
            if new_line_tup in current_line_set:
                continue
                
            current_lines[idx] = new_line
            current_line_set.discard(tuple(old_line))
            current_line_set.add(new_line_tup)
            
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                current_line_set.discard(new_line_tup)
                current_line_set.add(tuple(old_line))
                no_improve += 1
                
        elif move_type < 0.80:
            # Swap a point - try multiple candidates and pick best
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            best_new_I = current_I
            best_new_pt = None
            for _ in range(5):
                cand_idx = random.randint(0, n_cand_pts-1)
                new_pt = candidate_points[cand_idx].copy()
                
                if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                    continue
                
                current_points[pt_idx] = new_pt
                trial_I = count_incidences(current_points, current_lines)
                if trial_I > best_new_I:
                    best_new_I = trial_I
                    best_new_pt = new_pt.copy()
                current_points[pt_idx] = old_pt
            
            if best_new_pt is not None:
                current_points[pt_idx] = best_new_pt
                new_I = best_new_I
            else:
                new_I = current_I
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in seen:
                    seen.add(tuple(nl))
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    current_line_set = set(tuple(l) for l in current_lines)
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points - larger grid for more diversity
    candidate_points = generate_candidate_points(size=50, extra=2000)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # 1. Run Structured Search first to find a strong baseline
    if time.time() < deadline - 100:
        struct_lines, struct_points, struct_I = structured_search(candidate_points, time_limit=45)
        if struct_lines is not None and struct_I > best_I_global:
            best_I_global = struct_I
            best_points_global = struct_points.copy()
            best_lines_global = struct_lines.copy()
            
            # Refine the structured result with local search
            if time.time() < deadline - 50:
                # Create a pool of structured lines for the local search to sample from
                pool_lines = []
                for c in range(-40, 41):
                    pool_lines.append(np.array([1, 0, -c], dtype=np.int64))
                    pool_lines.append(np.array([0, 1, -c], dtype=np.int64))
                for c in range(-80, 81):
                    pool_lines.append(np.array([1, -1, -c], dtype=np.int64))
                    pool_lines.append(np.array([1, 1, -c], dtype=np.int64))
                
                bl, bp, bi = improve_configuration(struct_lines, candidate_points, 
                                                   time_limit=min(150, deadline - time.time() - 40), 
                                                   seed=100, 
                                                   structured_pool=pool_lines)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()

    # 2. Try other initial configurations if time permits
    for config_id in range(4):
        if time.time() > deadline - 60:
            break
            
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
            
        if time.time() < deadline - 70:
            remaining_time = min(80, deadline - time.time() - 50)
            if remaining_time > 15:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=200 + config_id * 17)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final refinement: do one more round of greedy line selection on best points
    if best_points_global is not None and time.time() < deadline - 40:
        refined_lines = greedy_select_lines_from_points(best_points_global, 27)
        unique_refined = []
        seen = set()
        for l in refined_lines:
            nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nl and tuple(nl) not in seen:
                seen.add(tuple(nl))
                unique_refined.append(nl)
        if len(unique_refined) >= 27:
            refined_lines = np.array(unique_refined[:27], dtype=np.int64)
            refined_points, _ = get_best_points_for_lines(refined_lines, candidate_points)
            refined_I = count_incidences(refined_points, refined_lines)
            if refined_I > best_I_global:
                best_I_global = refined_I
                best_points_global = refined_points.copy()
                best_lines_global = refined_lines.copy()
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    seen = set()
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and tuple(n_line) not in seen:
            seen.add(tuple(n_line))
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
