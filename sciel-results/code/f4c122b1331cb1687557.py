"""Combined approach: Structured initialization + Local Search/Hill Climbing."""

import time
import random
import math
from collections import Counter
import numpy as np

def _doubled_area_det(p1, p2, p3):
    return abs(
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    )

# Precompute triple indices once (i < j < k for n=30)
_n_points = 30
_triple_indices = None

def _get_triple_indices():
    global _triple_indices
    if _triple_indices is None:
        i_idx, j_idx, k_idx = np.indices((_n_points, _n_points, _n_points))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        _triple_indices = (i_idx[mask], j_idx[mask], k_idx[mask])
    return _triple_indices

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    ti, tj, tk = _get_triple_indices()
    
    # Extract coordinates using precomputed indices
    xi = pts[ti, 0]
    yi = pts[ti, 1]
    xj = pts[tj, 0]
    yj = pts[tj, 1]
    xk = pts[tk, 0]
    yk = pts[tk, 1]
    
    # Compute doubled areas
    areas = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Filter zero areas
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Count multiplicities
    vals, counts = np.unique(areas, return_counts=True)
    return int(counts.max())

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_diamond = []
    for x, y, d in diamond_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_diamond.append((x, y))
            if len(cand_diamond) == 30:
                break
    candidates.append(cand_diamond)

    # 3. Circle Shape (Euclidean distance)
    circle_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            circle_pts.append((x, y, x*x + y*y))
    circle_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_circle = []
    for x, y, d in circle_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_circle.append((x, y))
            if len(cand_circle) == 30:
                break
    candidates.append(cand_circle)
    
    # 4. Lattice patterns with different bases
    # Hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 2 + (row % 2)
            y = row
            hex_pts.append((x, y))
    if len(hex_pts) == 30:
        cx = 4
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # Triangular lattice
    tri_pts = []
    for row in range(6):
        for col in range(5):
            x = col
            y = row
            tri_pts.append((x, y))
    if len(tri_pts) == 30:
        cx = 2
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Diagonal line patterns
    for offset in range(-5, 6, 2):
        diag_pts = []
        for i in range(30):
            x = i - 15
            y = i + offset
            if -30 <= x <= 30 and -30 <= y <= 30:
                diag_pts.append((x, y))
        if len(diag_pts) == 30:
            candidates.append(diag_pts)
    
    # 6. Parabolic patterns
    for a in [1, 2, -1, -2]:
        parab_pts = []
        for i in range(-15, 15):
            x = i
            y = a * i * i // 10  # Scale to stay in bounds
            if -30 <= x <= 30 and -30 <= y <= 30:
                parab_pts.append((x, y))
        if len(parab_pts) == 30:
            candidates.append(parab_pts)
    
    # 7. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 8. More structured random: pick from small grid with some randomness
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 9. Random from larger grid
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Initialize with diverse candidates
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        
    # 2. Local Search / Hill Climbing with improved strategy
    current_points = best_points[:]
    current_score = best_score
    occupied = set(current_points)
    
    # Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Generate a single random neighbor with adaptive radius
        # Bias towards small moves, but allow larger moves when stuck
        if no_improve_count > max_no_improve // 2:
            # More exploration when stuck
            radius = random.choice([1, 2, 3, 4, 5, 7])
        else:
            # Mostly small moves
            if random.random() < 0.8:
                radius = 1
            else:
                radius = random.choice([2, 3, 4])
        
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        
        if dx == 0 and dy == 0:
            continue
            
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30 and new_p not in occupied:
            # Perform move
            occupied.remove(old_p)
            occupied.add(new_p)
            current_points[idx] = new_p
            
            new_score = _max_multiplicity(current_points)
            
            delta = new_score - current_score
            
            if delta > 0:
                # Accept and update best
                current_score = new_score
                best_score = new_score
                best_points = current_points[:]
                no_improve_count = 0
            elif delta == 0:
                # Accept (neutral move)
                current_score = new_score
                no_improve_count += 1
            else:
                # Simulated annealing: accept with probability exp(-|delta|/T)
                prob = math.exp(-abs(delta) / T)
                if random.random() < prob:
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Revert
                    occupied.remove(new_p)
                    occupied.add(old_p)
                    current_points[idx] = old_p
                    no_improve_count += 1
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Restart if stuck or cooled down too much
            if T <= T_min * 1.1 or no_improve_count > max_no_improve:
                # Restart from best
                current_points = best_points[:]
                current_score = best_score
                occupied = set(current_points)
                T = 1.0
                no_improve_count = 0
    
    return best_points
