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
    for c in range(-10, 11):
        lines_pool.add(_normalize_line(1, 0, -c))
        lines_pool.add(_normalize_line(0, 1, -c))
        
    # Diagonals: x + y = c, x - y = c
    for c in range(-20, 21):
        lines_pool.add(_normalize_line(1, 1, -c))
        lines_pool.add(_normalize_line(1, -1, -c))
        
    # Slope 2, 1/2, -2, -1/2
    for c in range(-25, 26):
        lines_pool.add(_normalize_line(2, 1, -c))
        lines_pool.add(_normalize_line(1, 2, -c))
        lines_pool.add(_normalize_line(2, -1, -c))
        lines_pool.add(_normalize_line(1, -2, -c))
        
    # Slope 3, 1/3, -3, -1/3
    for c in range(-30, 31):
        lines_pool.add(_normalize_line(3, 1, -c))
        lines_pool.add(_normalize_line(1, 3, -c))
        lines_pool.add(_normalize_line(3, -1, -c))
        lines_pool.add(_normalize_line(1, -3, -c))
        
    # Slope 4, 1/4, -4, -1/4
    for c in range(-35, 36):
        lines_pool.add(_normalize_line(4, 1, -c))
        lines_pool.add(_normalize_line(1, 4, -c))
        lines_pool.add(_normalize_line(4, -1, -c))
        lines_pool.add(_normalize_line(1, -4, -c))
        
    # Some steeper slopes
    for c in range(-40, 41):
        lines_pool.add(_normalize_line(5, 1, -c))
        lines_pool.add(_normalize_line(1, 5, -c))
        
    # Remove degenerate lines
    lines_pool.discard((0, 0, 0))
    
    return list(lines_pool)

def _evaluate_line_set(lines_set, num_points=54):
    """
    Given a set of lines, find the best num_points and return (score, points).
    Uses vectorized operations for speed.
    """
    if not lines_set:
        return 0, []
    
    ls = list(lines_set)
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
                # Bounding box to avoid huge coordinates
                if -30 <= x <= 30 and -30 <= y <= 30:
                    candidates.add((x, y))
                    
    if not candidates:
        return 0, []
        
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
    
    # Sort by degree descending
    sorted_indices = np.argsort(-point_degrees)
    
    # We want exactly num_points.
    selected_indices = sorted_indices[:num_points]
    selected_points = [cand_list[i] for i in selected_indices]
    score = int(point_degrees[selected_indices].sum())
    
    return score, selected_points

def _modularity_line_clustering(lines_set, points_set, resolution=1.0):
    """
    Use modularity maximization to cluster lines that share high point-overlap.
    Returns clusters as a list of sets of line indices.
    """
    ls = list(lines_set)
    ps = list(points_set)
    n_l = len(ls)
    n_p = len(ps)
    
    if n_l < 2 or n_p < 2:
        return [[i] for i in range(n_l)]
    
    # Build line-line similarity based on shared points
    # First, compute point-line incidence
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
    for _ in range(20):  # Limit iterations for speed
        best_gain = -np.inf
        best_pair = None
        
        for i in range(n_l):
            ci = cluster_ids[i]
            for j in range(i + 1, n_l):
                cj = cluster_ids[j]
                if ci == cj:
                    continue
                
                # Compute modularity gain from merging ci and cj
                # Q_new - Q_old = 2 * (sum of B between ci and cj) - (deg_ci + deg_cj)^2 / (2*m) + deg_ci^2/(2*m) + deg_cj^2/(2*m)
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

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization using modularity-based clustering.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin
    
    pool = _build_rich_pool()
    pool_set = set(pool)
    
    # Initial selection: Start with a strong grid core
    initial_lines = set()
    for c in range(8):
        initial_lines.add(_normalize_line(1, 0, -c))  # x=0..7
        initial_lines.add(_normalize_line(0, 1, -c))  # y=0..7
    
    # Add 11 more lines from pool to reach 27
    remaining_pool = [l for l in pool if l not in initial_lines]
    random.shuffle(remaining_pool)
    initial_lines.update(remaining_pool[:11])
    
    current_lines = set(initial_lines)
    
    # Evaluate current
    current_score, current_points = _evaluate_line_set(current_lines)
    best_score = current_score
    best_lines = set(current_lines)
    best_points = list(current_points)
    
    # Ensure we have exactly 54 points for the best set
    if len(best_points) < 54:
        used = set(best_points)
        while len(best_points) < 54:
            p = (random.randint(-15, 15), random.randint(-15, 15))
            if p not in used:
                best_points.append(p)
                used.add(p)
    else:
        best_points = best_points[:54]
        
    # Phase 1: Modularity-based clustering and replacement
    phase1_deadline = start_time + 180  # 3 minutes for clustering phase
    while time.time() < phase1_deadline:
        # Get current points (top 54 by degree)
        _, curr_pts = _evaluate_line_set(current_lines)
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
                        new_score, new_points = _evaluate_line_set(new_set)
                        
                        if new_score > current_score:
                            current_lines = new_set
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_lines = set(new_set)
                                best_points = list(new_points)
                            break
    
    # Phase 2: Local search with swaps
    phase2_deadline = deadline
    while time.time() < phase2_deadline:
        # Pick a line to remove
        l_out = random.choice(list(current_lines))
        # Pick a line to add
        candidates = list(pool_set - current_lines)
        if not candidates:
            break
        l_in = random.choice(candidates)
        
        # Try swap
        new_set = (current_lines - {l_out}) | {l_in}
        new_score, new_points = _evaluate_line_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_lines = set(new_set)
            best_points = list(new_points)
            current_lines = set(new_set)
            current_score = new_score
        elif new_score > current_score:
            current_lines = set(new_set)
            current_score = new_score
        
        # Occasionally try a 2-swap for diversity
        if random.random() < 0.1:
            l_out2 = random.choice(list(current_lines))
            candidates2 = list(pool_set - current_lines)
            if candidates2:
                l_in2 = random.choice(candidates2)
                new_set2 = (current_lines - {l_out2}) | {l_in2}
                new_score2, new_points2 = _evaluate_line_set(new_set2)
                
                if new_score2 > best_score:
                    best_score = new_score2
                    best_lines = set(new_set2)
                    best_points = list(new_points2)
                    current_lines = set(new_set2)
                    current_score = new_score2
    
    # Finalize best_points to exactly 54
    final_points = list(best_points)
    if len(final_points) < 54:
        used = set(final_points)
        while len(final_points) < 54:
            p = (random.randint(-20, 20), random.randint(-20, 20))
            if p not in used:
                final_points.append(p)
                used.add(p)
    else:
        final_points = final_points[:54]
        
    points_arr = np.array(final_points, dtype=np.int64)
    lines_arr = np.array(list(best_lines), dtype=np.int64)
    
    # Validation
    if points_arr.shape != (54, 2) or lines_arr.shape != (27, 3):
        # Fallback: Simple grid
        pts = []
        for i in range(7):
            for j in range(7):
                pts.append((i, j))
        # Need 54 points. 7x7=49. Add 5 more.
        pts.append((7, 0))
        pts.append((7, 1))
        pts.append((7, 2))
        pts.append((7, 3))
        pts.append((7, 4))
        points_arr = np.array(pts, dtype=np.int64)
        
        ls = []
        for i in range(8):
            ls.append((1, 0, -i))
        for j in range(7):
            ls.append((0, 1, -j))
        lines_arr = np.array(ls, dtype=np.int64)
        
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
