"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import defaultdict

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _evaluate(points):
    """Find the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    area_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    area_counts[a] += 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _greedy_select_for_area(pool, target_area, n_select=30):
    """
    Greedy: select n_select points from pool to maximize triples with area == target_area.
    Pool must be small (<=35) for this to be fast.
    """
    m = len(pool)
    if m < n_select or m > 35:
        return None
    
    selected = []
    remaining = list(range(m))
    
    for step in range(n_select):
        best_gain = -1
        best_idx = -1
        for idx in remaining:
            gain = 0
            sel = selected
            n_sel = len(sel)
            for i in range(n_sel):
                for j in range(i+1, n_sel):
                    a = _doubled_area(pool[idx], pool[sel[i]], pool[sel[j]])
                    if a == target_area:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    if len(selected) != n_select:
        return None
    
    return [pool[i] for i in selected]

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    
    Uses grid-based candidates plus greedy set-cover for area-targeted selection.
    """
    best_score = 0
    best_points = None
    
    # Strategy 1: Grid configurations
    grid_configs = [(6,5), (5,6), (7,5), (5,7), (8,4), (4,8), (10,3), (3,10), (15,2), (2,15)]
    
    for w, h in grid_configs:
        if w * h < 30:
            continue
        points = []
        x_start = -w // 2
        y_start = -h // 2
        for x in range(x_start, x_start + w):
            for y in range(y_start, y_start + h):
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
                if len(points) == 30:
                    break
            if len(points) == 30:
                break
        if len(points) != 30 or len(set(points)) != 30:
            continue
        score = _evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Strategy 2: Greedy for target areas with small pools (centered at origin)
    # Use a 7x7 pool = 49 points, too big. Use 6x6=36, still too big. Use 5x6=30 exactly.
    # Try pools of size 30-35
    pool_centers = [(0,0), (0,1), (1,0), (0,-1), (-1,0)]
    for cx, cy in pool_centers:
        pool = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x, y = cx + dx, cy + dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pool.append((x, y))
        # pool is 7x7=49, too big. Trim to 35
        if len(pool) > 35:
            pool = pool[:35]
        if len(pool) < 30:
            continue
        for target_area in [1, 2, 3, 4, 5, 6]:
            selected = _greedy_select_for_area(pool, target_area, 30)
            if selected is None or len(set(selected)) != 30:
                continue
            score = _evaluate(selected)
            if score > best_score:
                best_score = score
                best_points = selected
    
    # Strategy 3: Two parallel lines
    for dy in [1, 2, 3]:
        for x_start in range(-15, 0):
            points = []
            for i in range(15):
                x1 = x_start + i
                if -30 <= x1 <= 30:
                    points.append((x1, 0))
            for i in range(15):
                x2 = x_start + i + dy
                if -30 <= x2 <= 30:
                    points.append((x2, dy))
            if len(points) != 30 or len(set(points)) != 30:
                continue
            score = _evaluate(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
