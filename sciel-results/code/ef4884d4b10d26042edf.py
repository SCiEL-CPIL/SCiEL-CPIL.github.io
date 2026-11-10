# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

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
    Returns total incidences.
    """
    px = points[:, 0:1]
    py = points[:, 1:2]
    lA = lines[:, 0:1].T
    lB = lines[:, 1:2].T
    lC = lines[:, 2:3].T
    
    vals = px * lA + py * lB + lC
    incs = np.sum(vals == 0)
    return incs

def _build_rich_pool():
    """
    Build a rich pool of candidate lines with integer coefficients.
    Include axis-aligned, diagonal, and various slope lines.
    """
    lines_pool = set()
    
    # Axis aligned: x = c, y = c
    for c in range(-15, 16):
        lines_pool.add(_normalize_line(1, 0, -c))
        lines_pool.add(_normalize_line(0, 1, -c))
        
    # Diagonals: x + y = c, x - y = c
    for c in range(-25, 26):
        lines_pool.add(_normalize_line(1, 1, -c))
        lines_pool.add(_normalize_line(1, -1, -c))
        
    # Slope 2, 1/2, -2, -1/2
    for c in range(-30, 31):
        lines_pool.add(_normalize_line(2, 1, -c))
        lines_pool.add(_normalize_line(1, 2, -c))
        lines_pool.add(_normalize_line(2, -1, -c))
        lines_pool.add(_normalize_line(1, -2, -c))
        
    # Slope 3, 1/3, -3, -1/3
    for c in range(-35, 36):
        lines_pool.add(_normalize_line(3, 1, -c))
        lines_pool.add(_normalize_line(1, 3, -c))
        lines_pool.add(_normalize_line(3, -1, -c))
        lines_pool.add(_normalize_line(1, -3, -c))
        
    # Slope 4, 1/4, -4, -1/4
    for c in range(-40, 41):
        lines_pool.add(_normalize_line(4, 1, -c))
        lines_pool.add(_normalize_line(1, 4, -c))
        lines_pool.add(_normalize_line(4, -1, -c))
        lines_pool.add(_normalize_line(1, -4, -c))
        
    # Some steeper slopes
    for c in range(-45, 46):
        lines_pool.add(_normalize_line(5, 1, -c))
        lines_pool.add(_normalize_line(1, 5, -c))
        lines_pool.add(_normalize_line(5, -1, -c))
        lines_pool.add(_normalize_line(1, -5, -c))
        
    # Remove degenerate lines
    lines_pool.discard((0, 0, 0))
    
    return list(lines_pool)

def _fast_evaluate(points_arr, lines_arr):
    """
    Fast vectorized evaluation of incidences.
    points_arr: (N, 2) int64 array
    lines_arr: (M, 3) int64 array
    Returns (total_incidences, point_degrees, line_degrees)
    """
    if len(points_arr) == 0 or len(lines_arr) == 0:
        return 0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    px = points_arr[:, 0:1]
    py = points_arr[:, 1:2]
    lA = lines_arr[:, 0:1].T
    lB = lines_arr[:, 1:2].T
    lC = lines_arr[:, 2:3].T
    
    vals = px * lA + py * lB + lC
    inc_matrix = (vals == 0).astype(np.int64)
    
    point_degrees = inc_matrix.sum(axis=1)
    line_degrees = inc_matrix.sum(axis=0)
    total = int(inc_matrix.sum())
    
    return total, point_degrees, line_degrees

def _find_best_points_for_lines(lines, num_points=54, max_coord=40):
    """
    Find the best num_points integer points for the given lines.
    Returns (score, points) where points is a list of (x, y) tuples.
    """
    if not lines:
        return 0, []
    
    ls = list(lines)
    n_l = len(ls)
    
    # Find all integer intersection points
    candidates = set()
    for i in range(n_l):
        A1, B1, C1 = ls[i]
        for j in range(i + 1, n_l):
            A2, B2, C2 = ls[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                    candidates.add((x, y))
                    
    # If not enough candidates, add some grid points
    if len(candidates) < num_points:
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if (x, y) not in candidates:
                    candidates.add((x, y))
                if len(candidates) >= num_points:
                    break
            if len(candidates) >= num_points:
                break
    
    if not candidates:
        # Fallback: use simple grid
        pts = []
        for i in range(10):
            for j in range(10):
                pts.append((i, j))
        return 0, pts[:num_points]
        
    # Calculate degree for each candidate using vectorized ops
    cand_list = list(candidates)
    cand_arr = np.array(cand_list, dtype=np.int64)
    
    # Build incidence matrix: (n_points, n_lines)
    px = cand_arr[:, 0:1]
    py = cand_arr[:, 1:2]
    lA = np.array([l[0] for l in ls], dtype=np.int64).reshape(1, -1)
    lB = np.array([l[1] for l in ls], dtype=np.int64).reshape(1, -1)
    lC = np.array([l[2] for l in ls], dtype=np.int64).reshape(1, -1)
    
    vals = px * lA + py * lB + lC
    inc_matrix = (vals == 0).astype(np.int64)
    
    # Degree of each point
    point_degrees = inc_matrix.sum(axis=1)
    
    # Sort by degree descending, then by coordinate for tie-breaking
    sorted_indices = np.argsort(-point_degrees, kind='stable')
    
    # We want exactly num_points.
    selected_indices = sorted_indices[:num_points]
    selected_points = [cand_list[i] for i in selected_indices]
    score = int(point_degrees[selected_indices].sum())
    
    return score, selected_points

def _modularity_line_clustering(lines_set, points_set, resolution=1.0):
    """
    Use modularity maximization to cluster lines that share high point-overlap.
    Returns clusters as a list of lists of line indices.
    """
    ls = list(lines_set)
    ps = list(points_set)
    n_l = len(ls)
    n_p = len(ps)
    
    if n_l < 2 or n_p < 2:
        return [[i] for i in range(n_l)]
    
    # Build line-line similarity based on shared points
    px = np.array([p[0] for p in ps], dtype=np.int64).reshape(-1, 1)
    py = np.array([p[1] for p in ps], dtype=np.int64).reshape(-1, 1)
    lA = np.array([l[0] for l in ls], dtype=np.int64).reshape(1, -1)
    lB = np.array([l[1] for l in ls], dtype=np.int64).reshape(1, -1)
    lC = np.array([l[2] for l in ls], dtype=np.int64).reshape(1, -1)
    
    vals = px * lA + py * lB + lC
    inc_pl = (vals == 0).astype(np.int64)  # (n_p, n_l)
    
    # Line-line co-occurrence: A[i,j] = number of points on both line i and line j
    A = inc_pl.T @ inc_pl  # (n_l, n_l)
    
    # Degrees of lines
    deg = A.sum(axis=1)
    m = A.sum() / 2
    
    if m == 0:
        return [[i] for i in range(n_l)]
    
    # Modularity matrix
    B = A - resolution * np.outer(deg, deg) / (2 * m)
    
    # Simple greedy modularity maximization
    clusters = {i: [i] for i in range(n_l)}
    cluster_ids = {i: i for i in range(n_l)}
    
    # Iteratively merge pairs that increase modularity most
    for _ in range(15):  # Limit iterations for speed
        best_gain = -np.inf
        best_pair = None
        
        for i in range(n_l):
            ci = cluster_ids[i]
            for j in range(i + 1, n_l):
                cj = cluster_ids[j]
                if ci == cj:
                    continue
                
                # Compute modularity gain from merging ci and cj
                ci_lines = clusters[ci]
                cj_lines = clusters[cj]
                
                cross = sum(B[i2, j2] for i2 in ci_lines for j2 in cj_lines)
                deg_ci = sum(deg[i2] for i2 in ci_lines)
                deg_cj = sum(deg[j2] for j2 in cj_lines)
                
                gain = 2 * cross - (deg_ci + deg_cj)**2 / (2 * m) + deg_ci**2 / (2 * m) + deg_cj**2 / (2 * m)
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (ci, cj)
        
        if best_gain <= 0 or best_pair is None:
            break
        
        ci, cj = best_pair
        # Merge cj into ci
        for node in clusters[cj]:
            clusters[ci].append(node)
            cluster_ids[node] = ci
        del clusters[cj]
    
    return list(clusters.values())

def _find_best_replacement_line(cluster_lines, all_lines, current_points):
    """
    Find a line from all_lines that best covers points not well-covered by cluster_lines.
    """
    if not cluster_lines or not all_lines:
        return None
    
    # Points covered by cluster lines
    covered = set()
    for p in current_points:
        x, y = p
        for l in cluster_lines:
            A, B, C = l
            if A * x + B * y + C == 0:
                covered.add(p)
                break
    
    # Uncovered points
    uncovered = [p for p in current_points if p not in covered]
    
    if not uncovered:
        return None
    
    # Find line that covers most uncovered points
    best_line = None
    best_count = 0
    
    for l in all_lines:
        if l in cluster_lines:
            continue
        A, B, C = l
        count = sum(1 for p in uncovered if A * p[0] + B * p[1] + C == 0)
        if count > best_count:
            best_count = count
            best_line = l
    
    return best_line

def _build_concentrated_initial(lines_pool):
    """
    Build an initial configuration with points concentrated on few lines.
    Uses a dense local structure for higher incidence density.
    """
    initial_lines = set()
    
    # Core grid: 5x5 = 25 lines
    for c in range(5):
        initial_lines.add(_normalize_line(1, 0, -c))  # x=0..4
        initial_lines.add(_normalize_line(0, 1, -c))  # y=0..4
    
    # Add 2 more lines to reach 27
    remaining = [l for l in lines_pool if l not in initial_lines]
    random.shuffle(remaining)
    initial_lines.update(remaining[:2])
    
    return initial_lines

def _simulate_annealing(current_lines, current_points, pool_set, start_temp, end_temp, n_iterations, deadline):
    """
    Simulated annealing that explores both point and line moves.
    Returns (best_score, best_lines, best_points)
    """
    best_score = 0
    best_lines = set(current_lines)
    best_points = list(current_points)
    
    current_score, _, _ = _fast_evaluate(
        np.array(current_points, dtype=np.int64),
        np.array(list(current_lines), dtype=np.int64)
    )
    
    if current_score > best_score:
        best_score = current_score
        best_lines = set(current_lines)
        best_points = list(current_points)
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max(n_iterations - 1, 1))
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Line move: swap a line
            pool_diff = list(pool_set - current_lines)
            if not pool_diff:
                continue
            
            l_out = random.choice(list(current_lines))
            l_in = random.choice(pool_diff)
            
            new_lines = (current_lines - {l_out}) | {l_in}
            
            # Re-optimize points for new lines
            new_score, new_points = _find_best_points_for_lines(new_lines)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
                current_lines = new_lines
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_lines = set(current_lines)
                    best_points = list(current_points)
                    
        elif move_type < 0.8:
            # Point move: move a point to a better location
            if len(current_points) < 54:
                continue
                
            idx = random.randint(0, len(current_points) - 1)
            old_point = current_points[idx]
            
            # Try moving to a random nearby integer point
            new_x = old_point[0] + random.randint(-3, 3)
            new_y = old_point[1] + random.randint(-3, 3)
            new_point = (new_x, new_y)
            
            # Check for duplicates
            if new_point in current_points:
                continue
            
            new_points = list(current_points)
            new_points[idx] = new_point
            
            new_score, _, _ = _fast_evaluate(
                np.array(new_points, dtype=np.int64),
                np.array(list(current_lines), dtype=np.int64)
            )
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_lines = set(current_lines)
                    best_points = list(current_points)
                    
        else:
            # Replace a low-degree point with a high-degree candidate
            if len(current_points) >= 54:
                _, point_degrees, _ = _fast_evaluate(
                    np.array(current_points, dtype=np.int64),
                    np.array(list(current_lines), dtype=np.int64)
                )
                
                # Find lowest degree point
                min_idx = np.argmin(point_degrees)
                old_point = current_points[min_idx]
                
                # Find best replacement from intersection points
                ls = list(current_lines)
                candidates = set()
                for i in range(len(ls)):
                    A1, B1, C1 = ls[i]
                    for j in range(i + 1, len(ls)):
                        A2, B2, C2 = ls[j]
                        det = A1 * B2 - A2 * B1
                        if det == 0:
                            continue
                        num_x = B1 * C2 - B2 * C1
                        num_y = A2 * C1 - A1 * C2
                        if num_x % det == 0 and num_y % det == 0:
                            x = num_x // det
                            y = num_y // det
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                candidates.add((x, y))
                
                # Remove current points from candidates
                candidates -= set(current_points)
                
                if candidates:
                    new_point = random.choice(list(candidates))
                    new_points = list(current_points)
                    new_points[min_idx] = new_point
                    
                    new_score, _, _ = _fast_evaluate(
                        np.array(new_points, dtype=np.int64),
                        np.array(list(current_lines), dtype=np.int64)
                    )
                    
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
                        current_points = new_points
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_lines = set(current_lines)
                            best_points = list(current_points)
    
    return best_score, best_lines, best_points

def _ensure_valid_output(points_list, lines_set, max_coord=40):
    """
    Ensure we have exactly 54 points and 27 lines, all valid.
    Returns (points_arr, lines_arr) as numpy arrays.
    """
    # Ensure we have exactly 27 lines
    lines_list = list(lines_set)
    if len(lines_list) > 27:
        lines_list = lines_list[:27]
    elif len(lines_list) < 27:
        # Add more lines from a default set
        default_lines = []
        for c in range(-20, 21):
            default_lines.append(_normalize_line(1, 0, -c))
            default_lines.append(_normalize_line(0, 1, -c))
        for l in default_lines:
            if len(lines_list) >= 27:
                break
            if l not in lines_list:
                lines_list.append(l)
    
    # Ensure we have exactly 54 points
    points_list = list(points_list)
    if len(points_list) > 54:
        points_list = points_list[:54]
    elif len(points_list) < 54:
        # Add more points from a grid
        used = set(points_list)
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if len(points_list) >= 54:
                    break
                p = (x, y)
                if p not in used:
                    points_list.append(p)
                    used.add(p)
            if len(points_list) >= 54:
                break
    
    # Convert to numpy arrays
    points_arr = np.array(points_list, dtype=np.int64)
    lines_arr = np.array(lines_list, dtype=np.int64)
    
    return points_arr, lines_arr

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization using modularity-based clustering
    and simulated annealing for point/line moves.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin
    
    pool = _build_rich_pool()
    pool_set = set(pool)
    
    # Initial selection: Concentrated grid core
    current_lines = _build_concentrated_initial(pool)
    
    # Ensure we have exactly 27 lines
    if len(current_lines) > 27:
        current_lines = set(list(current_lines)[:27])
    elif len(current_lines) < 27:
        remaining = [l for l in pool if l not in current_lines]
        random.shuffle(remaining)
        current_lines.update(remaining[:27 - len(current_lines)])
    
    # Evaluate current
    current_score, current_points = _find_best_points_for_lines(current_lines)
    best_score = current_score
    best_lines = set(current_lines)
    best_points = list(current_points)
    
    # Phase 1: Modularity-based clustering and replacement (shorter)
    phase1_deadline = start_time + 120
    while time.time() < phase1_deadline:
        # Get current points (top 54 by degree)
        _, curr_pts = _find_best_points_for_lines(current_lines)
        if len(curr_pts) < 54:
            break
        
        # Cluster lines using modularity
        clusters = _modularity_line_clustering(current_lines, curr_pts, resolution=0.5)
        
        # Find the largest cluster (most redundant lines)
        if len(clusters) >= 2:
            largest_cluster = max(clusters, key=len)
            if len(largest_cluster) >= 3:
                # Try to replace some lines in this cluster
                cluster_lines = [list(current_lines)[i] for i in largest_cluster]
                
                # Find best replacement for one line in the cluster
                replacement = _find_best_replacement_line(cluster_lines, pool_set - current_lines, curr_pts)
                
                if replacement is not None:
                    # Try removing one line from the cluster and adding the replacement
                    for l_out in cluster_lines:
                        new_set = (current_lines - {l_out}) | {replacement}
                        new_score, new_points = _find_best_points_for_lines(new_set)
                        
                        if new_score > current_score:
                            current_lines = new_set
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_lines = set(new_set)
                                best_points = list(new_points)
                            break
    
    # Phase 2: Simulated annealing (main optimization)
    phase2_deadline = deadline - 30  # Leave 30s for final polish
    best_score, best_lines, best_points = _simulate_annealing(
        current_lines, current_points, pool_set,
        start_temp=10.0, end_temp=0.1,
        n_iterations=5000,
        deadline=phase2_deadline
    )
    
    # Phase 3: Final local search polish
    polish_deadline = deadline
    polish_score = best_score
    polish_lines = best_lines
    polish_points = best_points
    
    for _ in range(200):
        if time.time() > polish_deadline:
            break
        
        # Try single line swaps
        l_out = random.choice(list(polish_lines))
        candidates = list(pool_set - polish_lines)
        if not candidates:
            break
        l_in = random.choice(candidates)
        
        new_set = (polish_lines - {l_out}) | {l_in}
        new_score, new_points = _find_best_points_for_lines(new_set)
        
        if new_score > polish_score:
            polish_score = new_score
            polish_lines = new_set
            polish_points = new_points
    
    # Finalize: ensure valid output
    points_arr, lines_arr = _ensure_valid_output(polish_points, polish_lines)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
