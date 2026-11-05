"""
Combined program: maximize the largest multiplicity of a single positive doubled area
among all non-degenerate triangles formed by 30 distinct integer points in [-30,30]^2.

Strategy:
1. Start with a grid-based construction (5x6 grid gives 30 points).
2. Compute the full area histogram to find the target area (most common non-zero doubled area).
3. Use simulated annealing with O(N^2) incremental delta updates for the target area.
4. Periodically re-evaluate the full histogram to detect if a different area has become dominant.
5. Accept moves based on simulated annealing (accept improvements, probabilistically accept
   neutral/worse moves to escape local optima).
6. Use both small local moves and large random jumps for exploration.
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
    counts = Counter()
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
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
    old_pt = points[idx]
    others = [points[j] for j in range(len(points)) if j != idx]
    
    delta = 0
    n_others = len(others)
    
    for j in range(n_others):
        xj, yj = others[j]
        for k in range(j + 1, n_others):
            xk, yk = others[k]
            
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
                
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Try several initial grid configurations
    best_pts = None
    best_count = 0
    target_area = 0
    
    configs = [
        (6, 5, -2, -2),
        (5, 6, -2, -2),
        (10, 3, -4, -1),
        (3, 10, -1, -4),
        (15, 2, -7, 0),
        (2, 15, 0, -7),
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
                best_pts = pts[:]
                
    if best_pts is None:
        # Fallback: random points
        random.seed(0)
        best_pts = []
        while len(best_pts) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in best_pts:
                best_pts.append(p)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    
    # Simulated Annealing Parameters
    T = 15.0
    cooling = 0.9997
    min_T = 0.1
    
    current_pts = best_pts[:]
    current_count = best_count
    current_target = target_area
    
    last_reval_time = start_time
    
    while time.time() < deadline:
        # Periodic re-evaluation to update target and check global best
        if time.time() - last_reval_time > 1.5:
            c, t = _full_score_and_target(current_pts)
            if c > best_count:
                best_count = c
                best_pts = current_pts[:]
            # Update target to the current max area
            current_target = t
            current_count = c
            last_reval_time = time.time()
            # Reheat slightly on re-eval to allow exploration
            T = max(T, 5.0)
        
        # Propose a move
        if random.random() < 0.6:
            # Small local move
            idx = random.randint(0, 29)
            orig = current_pts[idx]
            radius = 3
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts:
                            candidates.append((nx, ny))
            if not candidates:
                continue
            new_pt = random.choice(candidates)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Simulated annealing acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        else:
            # Large random jump
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) in current_pts:
                continue
            new_pt = (nx, ny)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Simulated annealing acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        
        T *= cooling
        if T < min_T:
            # Reheat to escape local optima
            T = 8.0

    # Final check
    c, t = _full_score_and_target(best_pts)
    if c > best_count:
        best_count = c
    
    return [list(p) for p in best_pts]
