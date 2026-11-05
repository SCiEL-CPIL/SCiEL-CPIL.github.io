"""
Rewritten program: maximize the largest multiplicity of a single positive doubled area
among all non-degenerate triangles formed by 30 distinct integer points in [-30,30]^2.

Strategy:
1. Use a lattice-based construction. Points on a small grid or arithmetic progression
   tend to produce many equal-area triangles.
2. The doubled area of triangle (a,b,c) is |det(b-a, c-a)|.
   For points on a line, area is 0 (excluded).
   For points forming a parallelogram or regular lattice, many triangles share the same area.

Key insight: If we take points from a k x m grid (with k*m = 30), the doubled areas
take values that are multiples of the grid cell area. By choosing the grid dimensions
and orientation carefully, we can maximize the count of triangles with the same doubled area.

A 5x6 grid gives 30 points. Let's try that and also try other arrangements.

Better idea: Use a rectangular grid with dimensions that give many equal-area triangles.
For a standard axis-aligned grid, the doubled area of a triangle with vertices at grid points
is determined by the determinant. The most common non-zero doubled area in a dense grid
tends to be the smallest positive one (e.g., 1 or 2 for unit grid).

Let's compute for the 5x6 grid and see what we get, then try to improve via local search.

Actually, let's think more carefully. The task says "maximize the largest multiplicity of a
single positive doubled area". So we want ONE area value that appears in as many triangles as possible.

For 30 points, total triangles = C(30,3) = 4060. We want to maximize the count for one specific area.

A well-known construction: take points on a circle or on a line (but line gives 0 area).
Take points forming a regular polygon? Integer coordinates constrain us.

Let's try: a 5x6 grid. Compute all triple areas, find the max multiplicity.
Then try local perturbations to improve.

Given the 90s time limit, I'll do:
1. Start with a 5x6 grid.
2. Try several grid orientations/sizes.
3. Do a limited local search (swap points) to improve the score.
"""

import time
import random
from collections import Counter
import numpy as np

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _make_sheared_grid(cols, rows, x0=0, y0=0, shear=1):
    """Generate a sheared grid: point (c,r) -> (x0 + c + shear*r, y0 + r)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c + shear * r, y0 + r))
    return pts

def _make_double_lattice(cols, rows, x0=0, y0=0):
    """Two interleaved grids offset by (0.5, 0.5) scaled to integers.
    For even cols: use two grids of cols/2 x rows offset by (1,1)."""
    pts = []
    hc = cols // 2
    for r in range(rows):
        for c in range(hc):
            pts.append((x0 + 2*c, y0 + 2*r))
            if 2*c + 1 < cols:
                pts.append((x0 + 2*c + 1, y0 + 2*r + 1))
    # Trim or pad to exactly 30
    while len(pts) > 30:
        pts.pop()
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    # Use numpy for vectorized area computation to speed up
    pts = np.array(points, dtype=np.int64)
    # For each triple, compute doubled area
    # We'll do it in chunks to avoid huge memory
    for i in range(n):
        xi, yi = int(pts[i, 0]), int(pts[i, 1])
        for j in range(i+1, n):
            xj, yj = int(pts[j, 0]), int(pts[j, 1])
            for k in range(j+1, n):
                xk, yk = int(pts[k, 0]), int(pts[k, 1])
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx] is replaced by new_pt.
    """
    n = len(points)
    old_pt = points[idx]
    ox, oy = old_pt[0], old_pt[1]
    nx, ny = new_pt[0], new_pt[1]
    delta = 0
    
    # Iterate over all pairs (j, k) not involving idx
    # There are C(29, 2) = 406 pairs
    for j in range(n):
        if j == idx: continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx: continue
            xk, yk = points[k]
            
            # Old area
            d_old = abs((xj - ox) * (yk - oy) - (yj - oy) * (xk - ox))
            # New area
            d_new = abs((xj - nx) * (yk - ny) - (yj - ny) * (xk - nx))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx1] and points[idx2] are swapped.
    """
    n = len(points)
    p1 = points[idx1]
    p2 = points[idx2]
    delta = 0
    
    # Triangles involving exactly one of idx1 or idx2
    # Case 1: triangle (idx1, j, k) where j,k != idx1,idx2
    # Case 2: triangle (idx2, j, k) where j,k != idx1,idx2
    # Case 3: triangle (idx1, idx2, j) where j != idx1,idx2
    
    # Collect indices other than idx1 and idx2
    others = [i for i in range(n) if i != idx1 and i != idx2]
    
    # Case 1: (idx1, j, k) for j < k in others
    for a in range(len(others)):
        j = others[a]
        for b in range(a+1, len(others)):
            k = others[b]
            xj, yj = points[j]
            xk, yk = points[k]
            
            # Old: (p1, pj, pk)
            d_old1 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            # New: (p2, pj, pk)
            d_new1 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            
            if d_old1 == target_area and d_old1 > 0:
                delta -= 1
            if d_new1 == target_area and d_new1 > 0:
                delta += 1
            
            # Old: (p2, pj, pk)
            d_old2 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            # New: (p1, pj, pk)
            d_new2 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            
            if d_old2 == target_area and d_old2 > 0:
                delta -= 1
            if d_new2 == target_area and d_new2 > 0:
                delta += 1
    
    # Case 3: (idx1, idx2, j) for j in others
    for j in others:
        xj, yj = points[j]
        # Old: (p1, p2, pj)
        d_old = abs((p2[0] - p1[0]) * (yj - p1[1]) - (p2[1] - p1[1]) * (xj - p1[0]))
        # New: (p2, p1, pj) -> same triangle, same area
        d_new = d_old  # Swapping two vertices doesn't change the triangle
        
        # No change for this case since the triangle is the same
        # Actually wait - swapping idx1 and idx2 means the set of points is the same,
        # so all triangles have the same areas. The only "change" is in labeling.
        # So delta should be 0 for the swap itself if we're just permuting indices.
        # But we're not permuting - we're swapping positions in the array.
        # The set of 30 points remains the same! So swap doesn't change anything.
    
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    random.seed(42)
    
    # Try multiple initial configurations including sheared grids and double lattices
    best_pts = None
    best_count = 0
    target_area = 0
    
    # Standard grids
    grid_configs = [
        (6, 5, -2, -2),
        (5, 6, -2, -2),
        (10, 3, -4, -1),
        (3, 10, -1, -4),
        (15, 2, -7, 0),
        (2, 15, 0, -7),
    ]
    
    for cols, rows, x0, y0 in grid_configs:
        if cols * rows != 30:
            continue
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts) and len(set(pts)) == 30:
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = list(pts)
    
    # Sheared grids
    shear_configs = [
        (6, 5, -4, -2, 1),
        (5, 6, -2, -4, 1),
        (6, 5, -4, -2, 2),
        (5, 6, -2, -4, 2),
        (10, 3, -6, -1, 1),
        (3, 10, -1, -6, 1),
    ]
    
    for cols, rows, x0, y0, shear in shear_configs:
        if cols * rows != 30:
            continue
        pts = _make_sheared_grid(cols, rows, x0, y0, shear)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts) and len(set(pts)) == 30:
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = list(pts)
    
    if best_pts is None:
        best_pts = _make_grid(6, 5, -2, -2)
        best_count, target_area = _full_score_and_target(best_pts)
    
    # Local search with hill climbing and occasional simulated annealing
    no_improve_counter = 0
    sa_temperature = 2.0
    sa_cooling = 0.995
    
    while time.time() < deadline:
        # Periodically re-evaluate to see if a different area has become dominant
        if no_improve_counter > 80:
            c, t = _full_score_and_target(best_pts)
            if c > best_count:
                best_count = c
                target_area = t
            no_improve_counter = 0
            continue

        # Choose move type: single-point move (80%) or swap (20%)
        move_type = random.random()
        
        if move_type < 0.80:
            # Single point move
            idx = random.randint(0, 29)
            orig = best_pts[idx]
            improved = False
            
            # Use adaptive radius
            if no_improve_counter < 30:
                radius = 2
            elif no_improve_counter < 80:
                radius = 3
            else:
                radius = 4
            
            # Try all nearby positions, pick best delta
            best_delta = 0
            best_move = None
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in best_pts:
                        continue
                    
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (nx, ny)
            
            if best_move is not None:
                # Simulated annealing: accept worse moves with probability based on temperature
                if best_delta > 0:
                    best_pts[idx] = best_move
                    best_count += best_delta
                    no_improve_counter = 0
                elif best_delta < 0 and random.random() < sa_temperature:
                    best_pts[idx] = best_move
                    best_count += best_delta
                    no_improve_counter = 0
                sa_temperature *= sa_cooling
        else:
            # Swap two points
            i1 = random.randint(0, 29)
            i2 = random.randint(0, 29)
            if i1 == i2:
                continue
            
            # Try swapping with a nearby point
            p1 = best_pts[i1]
            # Find a candidate to swap with: try moving p1 to where p2 is and vice versa
            # This is equivalent to just permuting, so instead try moving i1 to a new location
            # and see if it's better. Actually, let's do a proper 2-point move.
            
            # Move i1 to a nearby location
            improved = False
            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = p1[0] + dx, p1[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in best_pts:
                        continue
                    
                    delta = _delta_count_for_target(best_pts, i1, (nx, ny), target_area)
                    if delta > 0:
                        best_pts[i1] = (nx, ny)
                        best_count += delta
                        no_improve_counter = 0
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_counter += 1
                sa_temperature = max(0.1, sa_temperature * 1.01)  # Reheat slightly
    
    # Final evaluation to get the true best count
    final_count, final_target = _full_score_and_target(best_pts)
    
    return [list(p) for p in best_pts]
