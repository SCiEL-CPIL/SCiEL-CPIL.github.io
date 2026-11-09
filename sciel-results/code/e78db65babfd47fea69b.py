import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """Fast score using numpy vectorization."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    counts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            if j + 1 < n:
                xk = pts[j+1:, 0]
                yk = pts[j+1:, 1]
                d = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                pos = d[d > 0]
                if len(pos) > 0:
                    for val in pos:
                        val = int(val)
                        if val in counts:
                            counts[val] += 1
                        else:
                            counts[val] = 1
    if not counts:
        return 0
    return max(counts.values())

def _compute_area_counts(points):
    """Compute area counts for a set of points."""
    n = len(points)
    if n < 3:
        return {}
    area_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    if d in area_counts:
                        area_counts[d] += 1
                    else:
                        area_counts[d] = 1
    return area_counts

def _incremental_delta(points, idx, old_pt, new_pt):
    """
    Compute the change in area counts when moving point idx from old_pt to new_pt.
    Returns dict of area -> change in count.
    """
    n = len(points)
    old_x, old_y = old_pt
    new_x, new_y = new_pt
    
    delta = {}
    
    for j in range(n):
        if j == idx:
            continue
        for k in range(j + 1, n):
            if k == idx:
                continue
            xj, yj = points[j]
            xk, yk = points[k]
            
            d_old = abs(old_x * (yj - yk) + xj * (yk - old_y) + xk * (old_y - yj))
            d_new = abs(new_x * (yj - yk) + xj * (yk - new_y) + xk * (new_y - yj))
            
            if d_old > 0:
                delta[d_old] = delta.get(d_old, 0) - 1
            if d_new > 0:
                delta[d_new] = delta.get(d_new, 0) + 1
    
    return delta

def _structured_candidates():
    """Generate structured initial candidates."""
    candidates = []
    
    # 1. Two parallel lines of 15 points each
    candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])
    
    # 2. Three parallel lines of 10 points each
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)])
    
    # 3. Four parallel lines
    candidates.append([(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + [(x, 2) for x in range(7)] + [(x, 3) for x in range(7)])
    
    # 4. 6x5 Grid
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    
    # 5. 10x3 Grid
    candidates.append([(x, y) for x in range(10) for y in range(3)])
    
    # 6. Two lines with spacing 2
    candidates.append([(x, 0) for x in range(15)] + [(x, 2) for x in range(15)])
    
    # 7. Three lines with spacing 2
    candidates.append([(x, 0) for x in range(10)] + [(x, 2) for x in range(10)] + [(x, 4) for x in range(10)])
    
    # 8. 5x6 Grid
    candidates.append([(x, y) for x in range(5) for y in range(6)])
    
    return candidates

def _local_search(points, t0, deadline):
    """Local search with incremental scoring."""
    n = len(points)
    current_points = [p for p in points]
    current_score = _fast_score(current_points)
    area_counts = _compute_area_counts(current_points)
    
    no_improve_count = 0
    best_points = [p for p in current_points]
    best_score = current_score
    
    while time.time() - t0 < deadline:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - t0 > deadline:
                break
                
            x, y = current_points[idx]
            # Expanded neighborhood
            moves = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    moves.append((dx, dy))
            
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pt = (new_x, new_y)
                if new_pt in current_points[:idx] + current_points[idx+1:]:
                    continue
                
                delta = _incremental_delta(current_points, idx, (x, y), new_pt)
                
                new_counts = {}
                for area, count in area_counts.items():
                    new_count = count + delta.get(area, 0)
                    if new_count > 0:
                        new_counts[area] = new_count
                
                new_max = max(new_counts.values()) if new_counts else 0
                
                if new_max > current_score:
                    current_points[idx] = new_pt
                    area_counts = new_counts
                    current_score = new_max
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p for p in current_points]
                else:
                    no_improve_count += 1
        
        if not improved:
            break
        if no_improve_count > 100:
            break
    
    return best_points

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    
    candidates = _structured_candidates()
    
    best_points = None
    best_score = 0
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        
        s = _fast_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        return [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    current_points = _local_search(best_points, t0, DEADLINE)
    
    return current_points
