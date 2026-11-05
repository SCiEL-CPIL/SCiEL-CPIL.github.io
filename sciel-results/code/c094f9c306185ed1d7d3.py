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

def _full_score(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """Calculate change in count of triangles with target_area when points[idx] is replaced."""
    n = len(points)
    old_pt = points[idx]
    delta = 0
    for j in range(n):
        if j == idx: continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx: continue
            xk, yk = points[k]
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Generate diverse initial candidates
    candidates = []
    
    # Rectangular grids
    for cols in [2, 3, 5, 6, 10, 15]:
        rows = 30 // cols
        for x0 in range(-5, 6):
            for y0 in range(-5, 6):
                pts = [(x0 + c, y0 + r) for r in range(rows) for c in range(cols)]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Two parallel lines (15 points each)
    for gap in range(1, 8):
        for x0 in range(-5, 6):
            for y0 in range(-5, 6):
                pts = [(x0 + i, y0) for i in range(15)] + [(x0 + i, y0 + gap) for i in range(15)]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Three parallel lines (10 points each)
    for gap in range(1, 6):
        for x0 in range(-5, 6):
            for y0 in range(-5, 6):
                pts = ([(x0 + i, y0) for i in range(10)] +
                       [(x0 + i, y0 + gap) for i in range(10)] +
                       [(x0 + i, y0 + 2*gap) for i in range(10)])
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Evaluate all candidates
    best_count = 0
    best_target = 0
    best_pts = None
    for pts in candidates:
        c, t = _full_score(pts)
        if c > best_count:
            best_count = c
            best_target = t
            best_pts = list(pts)
    
    if best_pts is None:
        best_pts = [(i, 0) for i in range(30)]
        best_count, best_target = _full_score(best_pts)
    
    # Local search
    random.seed(42)
    no_improve = 0
    current_count = best_count
    current_target = best_target
    
    while time.time() < deadline:
        # Periodically re-evaluate target
        if no_improve > 80:
            c, t = _full_score(best_pts)
            if c > current_count:
                current_count = c
                current_target = t
            no_improve = 0
            continue
        
        idx = random.randint(0, 29)
        orig = best_pts[idx]
        improved = False
        
        radius = 2 if no_improve < 40 else 3
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = orig[0] + dx, orig[1] + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                if (nx, ny) in best_pts:
                    continue
                delta = _delta_count_for_target(best_pts, idx, (nx, ny), current_target)
                if delta > 0:
                    best_pts[idx] = (nx, ny)
                    current_count += delta
                    no_improve = 0
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
        
        # Random jumps to escape local optima
        if no_improve > 150 and random.random() < 0.15:
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) not in best_pts:
                delta = _delta_count_for_target(best_pts, idx, (nx, ny), current_target)
                if delta > 0 or (delta == 0 and random.random() < 0.08):
                    best_pts[idx] = (nx, ny)
                    current_count += delta
                    no_improve = 0
    
    return [list(p) for p in best_pts]
