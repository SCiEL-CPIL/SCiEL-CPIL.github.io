import numpy as np
import time
from math import gcd
from collections import defaultdict
from itertools import combinations

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(int).tolist()))) != 54:
        return False
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _incidence_count(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # Efficient check: A*x + B*y + C == 0
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _construct_projective_plane_config():
    """
    Constructs a configuration in PG(2, 3) with 27 lines and 54 points.
    In PG(2, 3), there are 40 points and 40 lines.
    We want to select a subset.
    
    Strategy:
    1. Start with the full set of 40 points P and 40 lines L.
    2. We need to remove 14 points and 13 lines to get 54 points (wait, 40 is less than 54).
    
    Correction: PG(2, 3) has (3^2+3+1) = 13 points? No.
    q=3. Number of points = q^2 + q + 1 = 9 + 3 + 1 = 13.
    Number of lines = 13.
    This is too small.
    
    Let's use PG(2, 5)?
    q=5. Points = 25 + 5 + 1 = 31. Lines = 31.
    Still too small.
    
    Let's use PG(2, 7)?
    q=7. Points = 49 + 7 + 1 = 57. Lines = 57.
    This is close!
    We have 57 points and 57 lines.
    We need 54 points and 27 lines.
    
    In PG(2, 7):
    Each line has q+1 = 8 points.
    Each point is on q+1 = 8 lines.
    Total incidences = 57 * 8 = 456.
    
    If we select all 57 points, we have 3 extra points to remove.
    If we select 27 lines, we have 30 lines to remove.
    
    To maximize incidences, we should keep lines that intersect at high-multiplicity points.
    In PG(2, 7), every pair of lines intersects at exactly one point.
    The "multiplicity" of a point is the number of lines passing through it.
    In the full plane, every point has multiplicity 8.
    
    If we remove lines, the multiplicity of points drops.
    We want to keep lines such that the remaining points (which are intersections of kept lines) have high multiplicity.
    
    Let's try a different geometry: Affine plane AG(2, 7) or similar?
    AG(2, 7) has 49 points.
    Lines in AG(2, 7) have 7 points.
    Total lines: 7 * (7+1) = 56 lines?
    In AG(2, q), there are q parallel classes of q lines each. Total q(q+1) lines.
    For q=7, 56 lines.
    Points: 49.
    
    We need 54 points. AG(2, 7) only has 49.
    We can add 5 points at infinity (from the 8 points in PG(2, 7) not in AG(2, 7)).
    
    Let's stick to the coordinate construction but use the structural insight:
    The current program uses a greedy search on a small pool.
    The new strategy: **Explicit Construction via Finite Geometry**.
    
    1. Generate all 57 lines of PG(2, 7).
    2. Represent them as integer triples (A, B, C) in a way that allows mapping to 2D integer coordinates.
       Standard embedding: 
       Points in PG(2, 7) are [X:Y:Z].
       Affine chart Z=1: (x, y) = (X/Z, Y/Z).
       Since we need integer coordinates, we can map [X:Y:Z] to (X, Y) in a larger grid?
       No, the lines are linear equations.
       
       Let's use the fact that the problem allows ANY integer coordinates.
       We can map the projective plane points to a grid.
       However, preserving the incidence structure of PG(2, 7) exactly in 2D Euclidean integer grid is impossible (Sylvester-Gallai theorem says any non-collinear set of real points has an ordinary line).
       Wait, the Sylvester-Gallai theorem applies to real points.
       If we have 54 points, there MUST be an ordinary line (a line containing exactly 2 points) UNLESS all points are collinear.
       So we cannot have a "projective plane" configuration in 2D Euclidean space.
       
       Therefore, we must abandon exact PG structure and look for **near-miss** configurations or use the **Sylvester-Gallai** constraint to our advantage.
       
       Actually, the maximum number of incidences for n points and m lines is bounded by Szemeredi-Trotter: O(nm + n^(2/3)m^(2/3) + m).
       For n=54, m=27:
       54*27 = 1458.
       54^(2/3) * 27^(2/3) ≈ 14.4 * 9.1 ≈ 131.
       Total ~ 1600.
       
       The current score is 2.2346.
       Seed score? The prompt says "1.0 equals the seed program".
       So the seed program likely has a score of 1.0.
       The current program has I such that I/Seed_I = 2.2346.
       
       Let's look at a simpler, highly structured configuration that is valid in 2D:
       **Grid with Diagonals**.
       
       Consider a 6x9 grid of points (54 points).
       Lines:
       6 vertical lines (each with 9 points) -> 54 incidences.
       9 horizontal lines (each with 6 points) -> 54 incidences.
       12 diagonal lines?
       In a 6x9 grid, diagonals have lengths 1, 2, ..., 6, ..., 1.
       Sum of lengths of all diagonals in one direction:
       For width w=6, height h=9.
       Number of diagonals = w+h-1 = 14.
       Lengths: 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1? No.
       Max length is min(w, h) = 6.
       Number of diagonals of length k:
       For k=1: 2*(6+9-1) - 2*6 = ?
       Let's just calculate.
       Total incidences on diagonals = sum of lengths.
       It's known that sum of lengths of all diagonals in a grid is w*h*(w+h-1)/3? No.
       
       Let's use a **Complete Bipartite** style or **Incident** structure.
       
       Alternative Structural Change: **Simulated Annealing with a Better Initial State and Move Space**.
       The current program's local search is limited.
       
       Let's try a **Genetic Algorithm** or **Hill Climbing** starting from a **Grid + Diagonals** configuration.
       
       Configuration:
       Points: 6x9 grid.
       Lines:
       - All 6 verticals.
       - All 9 horizontals.
       - All "long" diagonals (slope 1 and -1) that cover at least 3 points.
       
       Let's implement a robust local search that:
       1. Starts with a Grid + Diagonals config.
       2. Uses a larger candidate pool for lines (small coefficients).
       3. Uses a larger candidate pool for points (near current points).
       4. Runs for the full time budget.
    """
    pass

def _grid_diagonal_initialization():
    """
    Create a 6x9 grid of points.
    Create lines for all verticals, horizontals, and main diagonals.
    """
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
            
    lines = []
    # Verticals
    for x in range(w):
        lines.append((1, 0, -x))
    # Horizontals
    for y in range(h):
        lines.append((0, 1, -y))
        
    # Diagonals with slope 1 (x - y = c)
    # x - y ranges from -8 to 5
    for c in range(-h+1, w):
        # Check if this line hits the grid
        # x = y + c
        # 0 <= y+c < w => -c <= y < w-c
        # 0 <= y < h
        y_min = max(0, -c)
        y_max = min(h, w-c)
        if y_max > y_min:
            # Line x - y - c = 0
            lines.append((1, -1, -c))
            
    # Diagonals with slope -1 (x + y = c)
    # x + y ranges from 0 to w+h-2
    for c in range(0, w+h-1):
        # x = c - y
        # 0 <= c-y < w => c-w < y <= c
        # 0 <= y < h
        y_min = max(0, c-w+1)
        y_max = min(h, c+1)
        if y_max > y_min:
            # Line x + y - c = 0
            lines.append((1, 1, -c))
            
    # We have too many lines. We need exactly 27.
    # Current count: 6 + 9 + 14 + 14 = 43 lines.
    # We need to select the best 27.
    # Sort by number of incidences (length) descending.
    
    line_lengths = []
    pts_arr = np.array(points)
    for l in lines:
        A, B, C = l
        vals = A * pts_arr[:, 0] + B * pts_arr[:, 1] + C
        cnt = int(np.sum(vals == 0))
        line_lengths.append((cnt, l))
        
    line_lengths.sort(key=lambda x: -x[0])
    
    # Take top 27 unique lines
    selected_lines = []
    seen = set()
    for cnt, l in line_lengths:
        canon = _normalize_line(l[0], l[1], l[2])
        if canon not in seen:
            seen.add(canon)
            selected_lines.append(canon)
            if len(selected_lines) == 27:
                break
                
    return np.array(points, dtype=np.int64), np.array(selected_lines, dtype=np.int64)

def _local_search_improved(points, lines, deadline):
    """
    Improved local search with better move generation.
    """
    import random
    rng = random.Random(42)
    
    pts = points.copy()
    lns = lines.copy()
    
    n_pts = 54
    n_lines = 27
    
    # Precompute incidence
    def compute_inc(pts, lns):
        vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
        return int(np.sum(vals == 0))
    
    current_score = compute_inc(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Candidate lines: small coefficients
    cand_lines = []
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.append(canon)
    
    # Deduplicate cand_lines
    seen_l = set()
    unique_cand_lines = []
    for l in cand_lines:
        if l not in seen_l:
            seen_l.add(l)
            unique_cand_lines.append(l)
    cand_lines = unique_cand_lines
    
    # Candidate points: around current points
    def get_cand_points(pts):
        cands = set()
        for p in pts:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cands.add((int(p[0])+dx, int(p[1])+dy))
        return list(cands)
    
    T_start = 5.0
    T_end = 0.1
    iter_count = 0
    n_iters = 200000
    
    while time.time() < deadline - 5:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            
            # Generate new point from candidates
            cands = get_cand_points(pts)
            new_pos = cands[rng.randint(0, len(cands)-1)]
            
            if new_pos == old_pos:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    dup = True
                    break
            if dup:
                continue
            
            # Calculate delta
            delta = 0
            for j in range(n_lines):
                A, B, C = lns[j]
                was = (A * old_pos[0] + B * old_pos[1] + C == 0)
                now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            new_score = current_score + delta
            if new_score >= current_score or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                pts[i] = new_pos
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        else:
            # Move a line
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines)-1)]
            
            if new_line == old_canon:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_lines):
                if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                    dup = True
                    break
            if dup:
                continue
            
            # Calculate delta
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                was = (lns[j, 0] * pts[i, 0] + lns[j, 1] * pts[i, 1] + lns[j, 2] == 0)
                now = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            new_score = current_score + delta
            if new_score >= current_score or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                lns[j] = new_line
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    
    return best_pts, best_lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Initialize with Grid + Diagonals
    pts, lns = _grid_diagonal_initialization()
    
    # 2. Local Search
    if time.time() < deadline - 10:
        pts, lns = _local_search_improved(pts, lns, deadline - 5)
        
    # 3. Final Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        return pts, lines
        
    return pts, lns
