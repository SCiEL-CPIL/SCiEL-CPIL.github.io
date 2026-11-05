import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

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
        for k in range(j+1, n_others):
            xk, yk = others[k]
            
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area:
                delta -= 1
            if d_new == target_area:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx1] and points[idx2] are swapped.
    """
    p1 = points[idx1]
    p2 = points[idx2]
    
    others = [points[j] for j in range(len(points)) if j != idx1 and j != idx2]
    
    delta = 0
    n_others = len(others)
    
    for j in range(n_others):
        xj, yj = others[j]
        for k in range(j+1, n_others):
            xk, yk = others[k]
            
            # Old: triangles involving p1 or p2 (but not both, since they're distinct)
            # Triangles with p1, j, k
            d_old_1 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            # Triangles with p2, j, k
            d_old_2 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            
            # New: p1 and p2 swapped, so same triangles
            # Actually swapping doesn't change which points form triangles
            # The triangles are the same set, just relabeled
            # So delta for swap of two points is 0 for the count!
            # Wait, that's not right. Swapping positions means the points are in different locations.
            # But the set of 30 points is the same, just permuted.
            # So the multiset of triangle areas is identical.
            # Therefore, swapping two points in the list doesn't change the score.
            # This move is useless for changing the configuration.
            
            # I was confused. A "swap" in the context of moving points means:
            # Move point at idx1 to where idx2 was, and idx2 to where idx1 was.
            # But that's just a permutation of the same 30 points, so the score is unchanged.
            # 
            # What we really want is: replace point idx1 with a new point, and replace point idx2 with a new point.
            # Or: move point idx1 to a new location (which is what _delta_count_for_target does).
            #
            # So swap moves as I described them are useless. Let me remove this function.
            
    return 0

def _try_local_search(pts, target_area, deadline, current_count=None):
    """Perform local search to improve the count for a given target_area."""
    if current_count is None:
        current_count, _ = _full_score_and_target(pts)
    
    current_pts = pts[:]
    T = 5.0
    cooling = 0.999
    min_T = 0.05
    
    while time.time() < deadline:
        idx = random.randint(0, 29)
        orig = current_pts[idx]
        
        # Try a random perturbation
        radius = random.choice([1, 2, 3, 5, 8])
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
        delta = _delta_count_for_target(current_pts, idx, new_pt, target_area)
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_pts[idx] = new_pt
            current_count += delta
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return current_pts, current_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    random.seed(42)
    
    # Generate diverse initial configurations
    configs = []
    
    # Standard grids
    for cols in range(1, 31):
        if 30 % cols == 0:
            rows = 30 // cols
            for x0 in range(-15, 16, 5):
                for y0 in range(-15, 16, 5):
                    pts = _make_grid(cols, rows, x0, y0)
                    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                        configs.append(pts)
    
    # Also try some structured non-grid configurations
    # Points on a circle (approximate)
    for radius in [5, 8, 10, 12]:
        pts = []
        for i in range(30):
            angle = 2 * 3.14159265 * i / 30
            x = int(round(radius * np.cos(angle)))
            y = int(round(radius * np.sin(angle)))
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
        if len(pts) == 30:
            configs.append(pts)
    
    # Diagonal lines
    for offset in range(-10, 11, 5):
        pts = []
        for i in range(30):
            x = i - 15 + offset
            y = i // 3
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
        if len(pts) == 30:
            configs.append(pts)
    
    # Evaluate all configs and pick the best few
    scored_configs = []
    for cfg in configs:
        c, t = _full_score_and_target(cfg)
        scored_configs.append((c, t, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Take top 5 configs for local search
    top_configs = scored_configs[:5]
    
    best_pts = None
    best_count = 0
    best_target = 0
    
    # Allocate time for local search on each top config
    num_searches = len(top_configs)
    time_per_search = (deadline - start_time) / (num_searches + 1)
    
    for i, (c, t, cfg) in enumerate(top_configs):
        if time.time() > deadline - 5:
            break
        
        search_deadline = min(deadline - 5, start_time + time_per_search * (i + 1))
        
        # Try multiple target areas from this config
        # Get top 3 areas from the histogram
        counts = Counter()
        n = len(cfg)
        for a in range(n):
            xa, ya = cfg[a]
            for b in range(a+1, n):
                xb, yb = cfg[b]
                for c_idx in range(b+1, n):
                    xc, yc = cfg[c_idx]
                    d = abs((xb - xa) * (yc - ya) - (yb - ya) * (xc - xa))
                    if d > 0:
                        counts[d] += 1
        
        top_areas = [area for area, cnt in counts.most_common(3)]
        
        for target_area in top_areas:
            if time.time() > search_deadline - 2:
                break
            
            result_pts, result_count = _try_local_search(cfg, target_area, search_deadline, c)
            
            if result_count > best_count:
                best_count = result_count
                best_pts = result_pts[:]
                best_target = target_area
    
    # If no good result, use the best initial config
    if best_pts is None:
        best_pts = top_configs[0][2][:]
        best_count = top_configs[0][0]
        best_target = top_configs[0][1]
    
    # Final verification
    final_count, final_target = _full_score_and_target(best_pts)
    
    return [list(p) for p in best_pts]

