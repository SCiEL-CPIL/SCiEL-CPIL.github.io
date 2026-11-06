import numpy as np
import random
import time
from collections import Counter

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    """
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
    def compute_all_areas(pts):
        """
        Compute all triangle areas for a set of points.
        Returns:
            counts: Counter of areas
            best_area: area with max count
            best_count: max count
        """
        n = len(pts)
        if n < 3:
            return Counter(), 0, 0
            
        pts_arr = np.array(pts, dtype=np.int64)
        
        # Vectorized computation of all triangle areas
        # i < j < k
        # We can use np.triu_indices on a 30x30 matrix? No, that's for pairs.
        # For triples, we can loop i, and use vectorized pairs for j,k > i.
        
        counts = Counter()
        for i in range(n):
            pi = pts_arr[i]
            others = pts_arr[i+1:]
            if len(others) < 2:
                continue
            # Pairs in others
            # indices in 'others'
            n_o = len(others)
            j_idx, k_idx = np.triu_indices(n_o, k=1)
            xj = others[j_idx, 0]
            yj = others[j_idx, 1]
            xk = others[k_idx, 0]
            yk = others[k_idx, 1]
            
            xi, yi = pi
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            # Filter zeros
            non_zero_areas = areas[areas > 0]
            if len(non_zero_areas) > 0:
                counts.update(non_zero_areas.tolist())
                
        if not counts:
            return Counter(), 0, 0
            
        best_area, best_count = counts.most_common(1)[0]
        return counts, int(best_area), int(best_count)

    # --- Initialization ---
    
    candidates = []
    
    # Pattern 1: Two parallel rows
    for y0 in range(-10, 10):
        for gap in [1, 2, 3, 4, 5]:
            y1 = y0 + gap
            if -30 <= y1 <= 30:
                for x_start in range(-10, 10):
                    pts = []
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)
                        
    # Pattern 2: Three parallel rows
    for y0 in range(-10, 10):
        for gap in [1, 2]:
            y1 = y0 + gap
            y2 = y0 + 2 * gap
            if -30 <= y2 <= 30:
                for x_start in range(-10, 10):
                    pts = []
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y2))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)

    # Pattern 3: Grids
    for w in range(5, 8):
        for h in range(5, 8):
            if w * h >= 30:
                for ox in range(-10, 10):
                    for oy in range(-10, 10):
                        pts = []
                        for x in range(w):
                            for y in range(h):
                                px = ox + x
                                py = oy + y
                                if -30 <= px <= 30 and -30 <= py <= 30:
                                    pts.append((px, py))
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if len(set(pts)) == 30:
                                candidates.append(pts)

    if not candidates:
        candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    random.seed(42)
    best_score = 0
    best_points = None
    best_counts = None
    best_area = 0
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        counts, area, count = compute_all_areas(cand)
        if count > best_score:
            best_score = count
            best_points = cand[:]
            best_counts = counts
            best_area = area

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_counts, best_area, best_score = compute_all_areas(best_points)

    # --- Local Search ---
    
    current_counts = best_counts.copy()
    current_best_area = best_area
    current_best_count = best_score
    current_points = best_points[:]
    
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    point_set = set(current_points)
    
    # Helper to compute new max count given old counts, old areas, new areas
    def get_new_max_count(counts, old_areas, new_areas, current_max_count):
        # Identify changed areas
        changed_areas = set(old_areas.tolist()) | set(new_areas.tolist())
        
        # Calculate new counts for changed areas
        delta = Counter()
        for a in old_areas:
            if a > 0:
                delta[a] -= 1
        for a in new_areas:
            if a > 0:
                delta[a] += 1
                
        max_new_count = 0
        
        # 1. Check changed areas
        for a in changed_areas:
            if a == 0:
                continue
            new_c = counts.get(a, 0) + delta.get(a, 0)
            if new_c > max_new_count:
                max_new_count = new_c
                
        # 2. Check unchanged areas
        # The max of unchanged areas is either the current max (if the current best area is unchanged)
        # or the second max (if the current best area is changed and decreased)
        # We can just iterate through counts, skipping changed areas.
        # Since counts is not huge, this is fast.
        max_unchanged_count = 0
        for a, c in counts.items():
            if a not in changed_areas:
                if c > max_unchanged_count:
                    max_unchanged_count = c
                    
        return max(max_new_count, max_unchanged_count)

    # Local search loop
    no_improvement_counter = 0
    max_no_improvement = 200
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        point_order = list(range(30))
        random.shuffle(point_order)
        
        for i in point_order:
            if time.time() - start_time > DEADLINE:
                break
                
            old_point = current_points[i]
            
            # Precompute 'others' data
            others_idx = [j for j in range(30) if j != i]
            others_pts = np.array([current_points[j] for j in others_idx], dtype=np.int64)
            
            n_o = len(others_pts)
            if n_o < 2:
                continue
                
            j_idx, k_idx = np.triu_indices(n_o, k=1)
            xj = others_pts[j_idx, 0]
            yj = others_pts[j_idx, 1]
            xk = others_pts[k_idx, 0]
            yk = others_pts[k_idx, 1]
            
            # Compute old areas for point i
            x0_old, y0_old = old_point
            old_areas = np.abs(x0_old * (yj - yk) + xj * (yk - y0_old) + xk * (y0_old - yj))
            
            # Generate candidates
            candidates_to_try = []
            
            # 1. Local perturbations
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ != old_point and np_ not in point_set:
                            candidates_to_try.append(np_)
                            
            # 2. Random valid points
            for _ in range(10):
                np_ = random.choice(all_valid_points)
                if np_ != old_point and np_ not in point_set:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:20]:
                if time.time() - start_time > DEADLINE:
                    break
                
                x0, y0 = new_point
                new_areas = np.abs(x0 * (yj - yk) + xj * (yk - y0) + xk * (y0 - yj))
                
                # Check if this move improves the global max
                new_max_count = get_new_max_count(current_counts, old_areas, new_areas, current_best_count)
                
                if new_max_count > current_best_count:
                    # Accept move
                    # Update counts
                    for a in old_areas:
                        if a > 0:
                            current_counts[a] -= 1
                            if current_counts[a] == 0:
                                del current_counts[a]
                    for a in new_areas:
                        if a > 0:
                            current_counts[a] += 1
                            
                    # Update state
                    point_set.remove(old_point)
                    point_set.add(new_point)
                    current_points[i] = new_point
                    current_best_count = new_max_count
                    # Update best_area to the one achieving the new max
                    # We need to find which area has the new max count
                    # It's either in changed areas or unchanged.
                    # We can just find the most common in current_counts
                    # This is O(K), which is fine.
                    current_best_area, _ = current_counts.most_common(1)[0]
                    
                    improved = True
                    no_improvement_counter = 0
                    break # Move to next point
        
        if not improved:
            no_improvement_counter += 1
            if no_improvement_counter > max_no_improvement:
                break

    return current_points
