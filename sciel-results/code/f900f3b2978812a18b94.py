import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product
from scipy.optimize import linear_sum_assignment

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
    Search for the best 27 lines from the families:
    Vertical: (1, 0, -x)
    Horizontal: (0, 1, -y)
    Diag+: (1, -1, -c)  => x - y = c
    Diag-: (1, 1, -c)   => x + y = c
    
    Uses grid-aware scoring: for V+H lines, the optimal points are at their
    intersections, so we score based on how many intersection points fall in range.
    """
    start_time = time.time()
    
    # Define candidate pools for each family
    v_intercepts = list(range(-40, 41))
    h_intercepts = list(range(-40, 41))
    d_plus_intercepts = list(range(-80, 81))
    d_minus_intercepts = list(range(-80, 81))
    
    P = candidate_points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    # Precompute incidence counts for each candidate line
    all_candidates = []
    for c in v_intercepts:
        all_candidates.append((1, 0, -c, 'V', c))
    for c in h_intercepts:
        all_candidates.append((0, 1, -c, 'H', c))
    for c in d_plus_intercepts:
        all_candidates.append((1, -1, -c, 'D+', c))
    for c in d_minus_intercepts:
        all_candidates.append((1, 1, -c, 'D-', c))
    
    # Compute incidence count for each candidate line
    line_counts = np.zeros(len(all_candidates), dtype=np.int32)
    for i, (A, B, C, fam, c) in enumerate(all_candidates):
        vals = A * px + B * py + C
        line_counts[i] = np.sum(vals == 0)
    
    # For V+H combinations, compute intersection-based score
    # A V line at x=a and H line at y=b intersect at (a,b)
    # If (a,b) is in the candidate pool, it's a "good" intersection
    
    # Build lookup: which (x,y) pairs are in candidate pool
    point_set = set()
    for i in range(len(P)):
        point_set.add((int(P[i,0]), int(P[i,1])))
    
    # Score V and H lines by how many of their intersections with other lines are valid points
    v_scores = np.zeros(len(v_intercepts), dtype=np.float64)
    h_scores = np.zeros(len(h_intercepts), dtype=np.float64)
    
    for vi, vx in enumerate(v_intercepts):
        for hi, hy in enumerate(h_intercepts):
            if (vx, hy) in point_set:
                v_scores[vi] += 1
                h_scores[hi] += 1
    
    # Try multiple allocation strategies
    variations = [
        {'V': 9, 'H': 9, 'D+': 5, 'D-': 4},
        {'V': 10, 'H': 10, 'D+': 4, 'D-': 3},
        {'V': 8, 'H': 8, 'D+': 6, 'D-': 5},
        {'V': 9, 'H': 9, 'D+': 4, 'D-': 5},
        {'V': 9, 'H': 9, 'D+': 6, 'D-': 3},
        {'V': 11, 'H': 11, 'D+': 3, 'D-': 2},
        {'V': 7, 'H': 7, 'D+': 7, 'D-': 6},
    ]
    
    best_I = -1
    best_lines = None
    best_points = None
    
    # Map family to candidate indices
    v_indices = list(range(len(v_intercepts)))
    h_indices = list(range(len(v_intercepts), len(v_intercepts)+len(h_intercepts)))
    dp_indices = list(range(len(v_intercepts)+len(h_intercepts), len(v_intercepts)+len(h_intercepts)+len(d_plus_intercepts)))
    dm_indices = list(range(len(v_intercepts)+len(h_intercepts)+len(d_plus_intercepts), len(all_candidates)))
    
    for var in variations:
        if time.time() - start_time > time_limit:
            break
            
        nV = var.get('V', 0)
        nH = var.get('H', 0)
        nDP = var.get('D+', 0)
        nDM = var.get('D-', 0)
        
        if nV + nH + nDP + nDM != 27:
            continue
        
        # Select top V lines by score (intersection count)
        v_ranked = np.argsort(v_scores)[::-1][:nV]
        # Select top H lines by score
        h_ranked = np.argsort(h_scores)[::-1][:nH]
        # Select top D+ lines by incidence count
        dp_counts = line_counts[dp_indices]
        dp_ranked = np.argsort(dp_counts)[::-1][:nDP]
        # Select top D- lines by incidence count
        dm_counts = line_counts[dm_indices]
        dm_ranked = np.argsort(dm_counts)[::-1][:nDM]
        
        sel_idx = []
        sel_idx.extend([v_indices[i] for i in v_ranked])
        sel_idx.extend([h_indices[i] for i in h_ranked])
        sel_idx.extend([dp_indices[i] for i in dp_ranked])
        sel_idx.extend([dm_indices[i] for i in dm_ranked])
        
        if len(sel_idx) < 27:
            continue
            
        lines = np.array([all_candidates[i][:3] for i in sel_idx], dtype=np.int64)
        
        # Check for duplicates
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
        
        # Use intersection-aware point selection
        points, _ = get_best_points_intersection_aware(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I:
            best_I = I
            best_lines = lines.copy()
            best_points = points.copy()
            
    # Also try pure top-k by incidence (no family constraint)
    if time.time() - start_time < time_limit - 5:
        sorted_indices = np.argsort(line_counts)[::-1][:27]
        lines = np.array([all_candidates[i][:3] for i in sorted_indices], dtype=np.int64)
        unique_lines = []
        seen = set()
        for l in lines:
            nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nl and nl not in seen:
                seen.add(nl)
                unique_lines.append(nl)
        if len(unique_lines) >= 27:
            lines = np.array(unique_lines[:27], dtype=np.int64)
            points, _ = get_best_points_intersection_aware(lines, candidate_points)
            I = count_incidences(points, lines)
            if I > best_I:
                best_I = I
                best_lines = lines.copy()
                best_points = points.copy()
            
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

def compute_incidence_matrix(points, lines):
    """Compute the full (n_points, n_lines) incidence matrix (0/1)."""
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    return (vals == 0).astype(np.int8)

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    Uses intersection-aware scoring: points at line intersections get bonus.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    
    inc_matrix = compute_incidence_matrix(P, L)
    counts = inc_matrix.sum(axis=1)
    
    # Bonus for points that are intersections of multiple lines
    # A point on k lines contributes k to total, but if it's a "hub" 
    # (intersection of lines that otherwise don't share other points), it's more valuable
    # Simple heuristic: slight bonus for high-count points
    scores = counts.astype(np.float64)
    scores[counts >= 3] += 0.1  # small bonus for 3+ incidence points
    
    # Get top k indices
    top_indices = np.argsort(scores)[::-1][:k]
    best_points = candidate_points[top_indices]
    
    return best_points, int(counts[top_indices].max())

def get_best_points_intersection_aware(lines, candidate_points, k=54):
    """
    More sophisticated point selection that considers line intersection structure.
    Points that are intersections of pairs of lines get priority.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    n_lines = len(L)
    
    inc_matrix = compute_incidence_matrix(P, L)
    counts = inc_matrix.sum(axis=1)
    
    # For each pair of lines, find if they intersect at a candidate point
    # This gives extra weight to "structural" intersection points
    intersection_bonus = np.zeros(len(P), dtype=np.int32)
    
    # Only check pairs for efficiency (n_lines choose 2)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            # Points on both line i and line j
            both = (inc_matrix[:, i] == 1) & (inc_matrix[:, j] == 1)
            if both.any():
                intersection_bonus[both] += 1
    
    # Score: base count + bonus for being an intersection
    scores = counts.astype(np.float64) + 0.05 * intersection_bonus
    
    top_indices = np.argsort(scores)[::-1][:k]
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

def grid_based_search(v_range, h_range, dp_range, dm_range, candidate_points, time_limit=30):
    """
    Exhaustive search over V/H line selections using the grid structure.
    For a fixed set of V and H lines, the optimal 54 points are the top-scoring
    grid intersections. Then we add D+ and D- lines greedily.
    """
    start_time = time.time()
    best_I = -1
    best_lines = None
    best_points = None
    
    P = candidate_points.astype(np.int64)
    
    # Build point set for quick lookup
    point_set = set()
    for i in range(len(P)):
        point_set.add((int(P[i,0]), int(P[i,1])))
    
    v_vals = list(range(*v_range))
    h_vals = list(range(*h_range))
    dp_vals = list(range(*dp_range))
    dm_vals = list(range(*dm_range))
    
    # Precompute: for each V line, which H lines create valid intersection points
    # and what additional D+ / D- incidences those points have
    
    # Try a few (nV, nH) splits
    splits = [(9, 9), (10, 10), (8, 8), (11, 11)]
    
    for nV, nH in splits:
        if time.time() - start_time > time_limit:
            break
        
        nD = 27 - nV - nH
        if nD < 2:
            continue
        
        # Try selecting top nV verticals and top nH horizontals by "coverage"
        # Coverage of a V line = number of H lines such that (v, h) is in point_set
        v_cov = np.zeros(len(v_vals), dtype=np.int32)
        h_cov = np.zeros(len(h_vals), dtype=np.int32)
        
        for vi, v in enumerate(v_vals):
            for hi, h in enumerate(h_vals):
                if (v, h) in point_set:
                    v_cov[vi] += 1
                    h_cov[hi] += 1
        
        # Select top nV by coverage
        top_v = np.argsort(v_cov)[::-1][:nV]
        top_h = np.argsort(h_cov)[::-1][:nH]
        
        sel_v = [v_vals[i] for i in top_v]
        sel_h = [h_vals[i] for i in top_h]
        
        # Build the base lines
        base_lines = []
        for v in sel_v:
            base_lines.append((1, 0, -v))
        for h in sel_h:
            base_lines.append((0, 1, -h))
        
        # Compute incidence of each candidate D+ and D- line against grid points
        # that are at V/H intersections
        intersection_pts = [(v, h) for v in sel_v for h in sel_h if (v, h) in point_set]
        # Actually, let's just compute against all candidate points
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        
        dp_counts = []
        for c in dp_vals:
            vals = 1 * px + (-1) * py + (-c)
            dp_counts.append(int(np.sum(vals == 0)))
        
        dm_counts = []
        for c in dm_vals:
            vals = 1 * px + 1 * py + (-c)
            dm_counts.append(int(np.sum(vals == 0)))
        
        # Select top D+ and D- lines
        n_dp = nD // 2 + (nD % 2)
        n_dm = nD - n_dp
        
        top_dp = np.argsort(dp_counts)[::-1][:n_dp]
        top_dm = np.argsort(dm_counts)[::-1][:n_dm]
        
        for i in top_dp:
            base_lines.append((1, -1, -dp_vals[i]))
        for i in top_dm:
            base_lines.append((1, 1, -dm_vals[i]))
        
        lines = np.array(base_lines[:27], dtype=np.int64)
        
        # Check uniqueness
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
        points, _ = get_best_points_intersection_aware(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I:
            best_I = I
            best_lines = lines.copy()
            best_points = points.copy()
    
    return best_lines, best_points, best_I

def greedy_select_lines_from_points(points, n_lines=27, candidate_pool=None):
    """
    Greedily select lines that maximize total incidences with the given points.
    Uses marginal gain: each new line's value is the number of NEW incidences
    it creates (points not already covered by selected lines).
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
            dx = int(points[j, 0] - points[i, 0])
            dy = int(points[j, 1] - points[i, 1])
            A = dy
            B = -dx
            C = dx * int(points[i, 1]) - dy * int(points[i, 0])
            nl = normalize_line(A, B, C)
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    # Pre-compute incidence vectors for all candidate lines
    line_vecs = []
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        line_vecs.append((vals == 0).astype(np.int8))
    
    # Greedy selection with marginal gain
    selected_lines = []
    selected_set = set()
    covered = np.zeros(n_pts, dtype=np.int8)  # points already on some selected line
    
    for _ in range(n_lines):
        best_line = None
        best_marginal = -1
        for idx, nl in enumerate(candidate_lines):
            if nl in selected_set:
                continue
            # Marginal gain = number of points on this line NOT already covered
            vec = line_vecs[idx]
            marginal = np.sum(vec & (~covered))
            # Also count total as tiebreaker
            total = np.sum(vec)
            if marginal > best_marginal or (marginal == best_marginal and total > best_marginal):
                best_marginal = marginal
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
        # Update coverage
        vec = line_vecs[candidate_lines.index(best_line)]
        covered = np.maximum(covered, vec)
    
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
    Improved: faster evaluation via precomputed incidence, better moves.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Precompute a set of current line tuples for fast duplicate check
    current_line_set = set(tuple(l) for l in current_lines)
    
    # Precompute incidence matrix for current state
    inc_matrix = compute_incidence_matrix(current_points, current_lines)
    
    while time.time() - start_time < time_limit:
        if no_improve > 200:
            T = max(T, 10.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = int(p2[0] - p1[0])
            dy = int(p2[1] - p1[1])
            A = dy
            B = -dx
            C = dx * int(p1[1]) - dy * int(p1[0])
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if new_line_tup in current_line_set:
                continue
                
            current_lines[idx] = np.array(new_line, dtype=np.int64)
            current_line_set.discard(tuple(old_line))
            current_line_set.add(new_line_tup)
            
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                inc_matrix = compute_incidence_matrix(current_points, current_lines)
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
                
        elif move_type < 0.55:
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
                inc_matrix = compute_incidence_matrix(current_points, current_lines)
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
                
        elif move_type < 0.85:
            # Swap a point - try multiple candidates and pick best
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            best_new_I = current_I
            best_new_pt = None
            for _ in range(8):
                cand_idx = random.randint(0, n_cand_pts-1)
                new_pt = candidate_points[cand_idx].copy()
                
                # Quick check: is this point already in use?
                if np.any(np.all(current_points == new_pt, axis=1)):
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
                    inc_matrix = compute_incidence_matrix(current_points, current_lines)
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
                                                   time_limit=min(90, deadline - time.time() - 40), 
                                                   seed=100, 
                                                   structured_pool=pool_lines)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()

    # 2. Try grid-based search with different ranges
    if time.time() < deadline - 80:
        gl, gp, gi = grid_based_search((-20, 21), (-20, 21), (-60, 61), (-60, 61), 
                                        candidate_points, time_limit=30)
        if gl is not None and gi > best_I_global:
            best_I_global = gi
            best_points_global = gp.copy()
            best_lines_global = gl.copy()
            
            # Refine with local search
            if time.time() < deadline - 40:
                pool_lines = []
                for c in range(-40, 41):
                    pool_lines.append(np.array([1, 0, -c], dtype=np.int64))
                    pool_lines.append(np.array([0, 1, -c], dtype=np.int64))
                for c in range(-80, 81):
                    pool_lines.append(np.array([1, -1, -c], dtype=np.int64))
                    pool_lines.append(np.array([1, 1, -c], dtype=np.int64))
                
                bl, bp, bi = improve_configuration(gl, candidate_points, 
                                                   time_limit=min(60, deadline - time.time() - 30), 
                                                   seed=200, 
                                                   structured_pool=pool_lines)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()

    # 3. Try other initial configurations if time permits
    for config_id in range(4):
        if time.time() > deadline - 40:
            break
            
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
            
        if time.time() < deadline - 50:
            remaining_time = min(80, deadline - time.time() - 40)
            if remaining_time > 15:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=200 + config_id * 17)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final refinement: do one more round of greedy line selection on best points
    if best_points_global is not None and time.time() < deadline - 20:
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
