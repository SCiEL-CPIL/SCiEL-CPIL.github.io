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
import numpy as np
from collections import Counter

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    pts_arr = np.array(points, dtype=np.int64)
    counts = Counter()
    for i in range(n):
        xi, yi = int(pts_arr[i][0]), int(pts_arr[i][1])
        for j in range(i+1, n):
            xj, yj = int(pts_arr[j][0]), int(pts_arr[j][1])
            for k in range(j+1, n):
                xk, yk = int(pts_arr[k][0]), int(pts_arr[k][1])
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
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            # New area
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx1] and points[idx2] are swapped.
    Triangles including both idx1 and idx2 are unchanged.
    Triangles including exactly one of them change.
    """
    n = len(points)
    p1 = points[idx1]
    p2 = points[idx2]
    delta = 0
    
    for j in range(n):
        if j == idx1 or j == idx2: continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx1 or k == idx2: continue
            xk, yk = points[k]
            
            # Triangle (idx1, j, k): old uses p1, new uses p2
            d_old = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            d_new = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
            
            # Triangle (idx2, j, k): old uses p2, new uses p1
            d_old2 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            d_new2 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            if d_old2 == target_area and d_old2 > 0:
                delta -= 1
            if d_new2 == target_area and d_new2 > 0:
                delta += 1
    
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Start with a 5x6 grid (30 points)
    best_pts = _make_grid(6, 5, -2, -2)
    
    # Evaluate initial score
    best_count, target_area = _full_score_and_target(best_pts)
    
    # Try a few different grid configurations to find a better starting point
    configs = [
        (6, 5, -2, -2),
        (5, 6, -2, -2),
        (10, 3, -4, -1),
        (3, 10, -1, -4),
        (15, 2, -7, 0),
        (2, 15, 0, -7),
        (1, 30, -14, 0),
        (30, 1, 0, -14),
    ]
    
    for cols, rows, x0, y0 in configs:
        if cols * rows != 30:
            continue
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = pts

    random.seed(42)
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Periodically re-evaluate to see if a different area has become dominant
        if no_improve_counter > 80:
            c, t = _full_score_and_target(best_pts)
            if c > best_count:
                best_count = c
                target_area = t
            no_improve_counter = 0
            continue

        # Choose move type: single point move or two-point swap
        use_swap = random.random() < 0.3
        
        improved = False
        
        if use_swap:
            # Two-point swap: try swapping two random points
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 != idx2:
                delta = _delta_count_for_swap(best_pts, idx1, idx2, target_area)
                if delta > 0:
                    best_pts[idx1], best_pts[idx2] = best_pts[idx2], best_pts[idx1]
                    best_count += delta
                    no_improve_counter = 0
                    improved = True
            
            if not improved:
                # Try moving one of the points to a nearby location
                idx = random.randint(0, 29)
                orig = best_pts[idx]
                radius = 3 if no_improve_counter < 60 else 5
                for dx in range(-radius, radius + 1):
                    if improved: break
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        if (nx, ny) in best_pts:
                            continue
                        delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                        if delta > 0:
                            best_pts[idx] = (nx, ny)
                            best_count += delta
                            no_improve_counter = 0
                            improved = True
                            break
        else:
            # Single point move
            idx = random.randint(0, 29)
            orig = best_pts[idx]
            radius = 3 if no_improve_counter < 60 else 5
            for dx in range(-radius, radius + 1):
                if improved: break
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in best_pts:
                        continue
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > 0:
                        best_pts[idx] = (nx, ny)
                        best_count += delta
                        no_improve_counter = 0
                        improved = True
                        break
        
        if not improved:
            no_improve_counter += 1
            
        # Occasionally try a random jump to escape local optima
        if no_improve_counter > 150 and random.random() < 0.1:
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) not in best_pts:
                delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                if delta > 0 or (delta == 0 and random.random() < 0.05):
                    best_pts[idx] = (nx, ny)
                    best_count += delta
                    no_improve_counter = 0

    return [list(p) for p in best_pts]
