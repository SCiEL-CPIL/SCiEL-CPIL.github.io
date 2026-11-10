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
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1] # (N, 1)
    py = points[:, 1:2] # (N, 1)
    lA = lines[:, 0:1].T # (1, M)
    lB = lines[:, 1:2].T # (1, M)
    lC = lines[:, 2:3].T # (1, M)
    
    vals = px * lA + py * lB + lC
    incs = np.sum(vals == 0)
    return int(incs)

def _generate_line_pool():
    """Generate a diverse pool of lines."""
    lines_pool = set()
    
    # Axis aligned
    for c in range(-15, 16):
        lines_pool.add(_normalize_line(1, 0, -c))
        lines_pool.add(_normalize_line(0, 1, -c))
        
    # Diagonals slope 1
    for c in range(-30, 31):
        lines_pool.add(_normalize_line(1, 1, -c))
        
    # Diagonals slope -1
    for c in range(-30, 31):
        lines_pool.add(_normalize_line(1, -1, -c))
        
    # Slope 2, 1/2, -2, -1/2
    for c in range(-40, 41):
        lines_pool.add(_normalize_line(2, 1, -c))
        lines_pool.add(_normalize_line(1, 2, -c))
        lines_pool.add(_normalize_line(2, -1, -c))
        lines_pool.add(_normalize_line(1, -2, -c))
        
    # Other slopes
    for c in range(-40, 41):
        lines_pool.add(_normalize_line(3, 1, -c))
        lines_pool.add(_normalize_line(1, 3, -c))
        lines_pool.add(_normalize_line(3, -1, -c))
        lines_pool.add(_normalize_line(1, -3, -c))
        
    return list(lines_pool)

def _intersections(lines_list):
    """Find all integer intersection points of the given lines."""
    candidates = set()
    n_l = len(lines_list)
    for i in range(n_l):
        A1, B1, C1 = lines_list[i]
        for j in range(i + 1, n_l):
            A2, B2, C2 = lines_list[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                # Bounding box to avoid huge coordinates
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.add((x, y))
    return candidates

def _pick_points_optimal(lines_list, num_points=54):
    """
    Given a set of lines, find the best num_points and return (score, points).
    Uses a greedy approach with local search for point selection.
    """
    if not lines_list:
        return 0, []
    
    candidates = _intersections(lines_list)
    
    if not candidates:
        return 0, []
    
    # Calculate degree for each candidate
    point_degrees = {}
    for p in candidates:
        x, y = p
        deg = 0
        for l in lines_list:
            A, B, C = l
            if A * x + B * y + C == 0:
                deg += 1
        point_degrees[p] = deg
        
    # Sort by degree descending
    sorted_items = sorted(point_degrees.items(), key=lambda kv: kv[1], reverse=True)
    
    # We want exactly num_points.
    selected_points = [p for p, d in sorted_items[:num_points]]
    score = sum(d for p, d in sorted_items[:num_points])
    
    # Local search for point selection
    # Try swapping selected points with unselected ones
    selected_set = set(selected_points)
    unselected = [p for p, d in sorted_items[num_points:]]
    
    improved = True
    while improved:
        improved = False
        for i in range(len(selected_points)):
            p_i = selected_points[i]
            deg_i = point_degrees[p_i]
            # Try swapping with an unselected point
            random.shuffle(unselected)
            for p_j in unselected[:5]:  # Only check a few
                deg_j = point_degrees[p_j]
                if deg_j > deg_i:
                    # Swap
                    selected_points[i] = p_j
                    selected_set.remove(p_i)
                    selected_set.add(p_j)
                    unselected.remove(p_j)
                    unselected.append(p_i)
                    score += (deg_j - deg_i)
                    improved = True
                    break
                    
    return score, selected_points

def _score(lines_set):
    """Score a set of lines by finding optimal points."""
    lines_list = list(lines_set)
    score, _ = _pick_points_optimal(lines_list)
    return score

def _initial_lines(pool):
    """Create an initial set of lines."""
    initial_lines = set()
    
    # Start with a 7x7 grid core
    for c in range(7):
        initial_lines.add(_normalize_line(1, 0, -c)) # x=0..6
        initial_lines.add(_normalize_line(0, 1, -c)) # y=0..6
        
    # Add 13 more lines from pool to reach 27
    remaining_pool = [l for l in pool if l not in initial_lines]
    random.shuffle(remaining_pool)
    initial_lines.update(remaining_pool[:13])
    
    return initial_lines

def _local_search(current_lines, pool, deadline, best_score, best_lines, best_points):
    """
    Perform local search to improve the line set.
    """
    pool_set = set(pool)
    current_score, current_points = _pick_points_optimal(list(current_lines))
    
    if current_score > best_score:
        best_score = current_score
        best_lines = set(current_lines)
        best_points = list(current_points)
        
    steps = 0
    while time.time() < deadline:
        # Pick a line to remove
        l_out = random.choice(list(current_lines))
        
        # Pick a line to add
        available = [l for l in pool_set if l not in current_lines]
        if not available:
            break
        l_in = random.choice(available)
        
        # Try swap
        new_set = (current_lines - {l_out}) | {l_in}
        new_score, new_points = _pick_points_optimal(list(new_set))
        
        if new_score > best_score:
            best_score = new_score
            best_lines = set(new_set)
            best_points = list(new_points)
            current_lines = set(new_set)
            current_score = new_score
        elif new_score > current_score:
            current_lines = set(new_set)
            current_score = new_score
            
        steps += 1
        if steps % 50 == 0:
            # Occasional restart from best
            if random.random() < 0.1:
                current_lines = set(best_lines)
                current_score = best_score
                
    return best_score, best_lines, best_points

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for safety
    
    random.seed(42)  # For reproducibility
    
    pool = _generate_line_pool()
    
    # Initial selection
    current_lines = _initial_lines(pool)
    
    best_score = 0
    best_lines = set()
    best_points = []
    
    # Run local search
    best_score, best_lines, best_points = _local_search(current_lines, pool, deadline, best_score, best_lines, best_points)
    
    # Finalize best_points to exactly 54
    final_points = list(best_points)
    if len(final_points) < 54:
        used = set(final_points)
        while len(final_points) < 54:
            p = (random.randint(-50, 50), random.randint(-50, 50))
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
