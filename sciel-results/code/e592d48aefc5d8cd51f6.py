import numpy as np
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Pre-compute triple indices
    n = 30
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    TRI_I, TRI_J, TRI_K = i_idx[mask], j_idx[mask], k_idx[mask]
    N_TRIPLES = len(TRI_I)
    
    def compute_all_areas(points):
        """Vectorized computation of all doubled areas."""
        pi = points[TRI_I]
        pj = points[TRI_J]
        pk = points[TRI_K]
        dx_ji = pj[:, 0] - pi[:, 0]
        dy_ji = pj[:, 1] - pi[:, 1]
        dx_ki = pk[:, 0] - pi[:, 0]
        dy_ki = pk[:, 1] - pi[:, 1]
        dets = dx_ji * dy_ki - dy_ji * dx_ki
        return np.abs(dets)
    
    def get_max_multiplicity(areas):
        """Return (max_count, most_frequent_area) ignoring zeros."""
        non_zero = areas[areas > 0]
        if len(non_zero) == 0:
            return 0, 0
        unique_areas, counts = np.unique(non_zero, return_counts=True)
        max_idx = np.argmax(counts)
        return counts[max_idx], unique_areas[max_idx]
    
    def count_target(areas, target):
        """Count occurrences of target in areas."""
        return int(np.sum(areas == target))
    
    def make_grid(rows, cols, spacing=2, ox=0, oy=0):
        """Create a grid of points."""
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) < 30:
                    pts.append((i * spacing + ox, j * spacing + oy))
        return np.array(pts, dtype=np.int32)
    
    def make_sheared_grid(rows, cols, spacing=2, shear=1, ox=0, oy=0):
        """Create a sheared grid."""
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) < 30:
                    x = i * spacing + j * shear + ox
                    y = j * spacing + oy
                    pts.append((x, y))
        return np.array(pts, dtype=np.int32)
    
    def make_clustered_random(seed=42):
        """Create clustered random points."""
        rng = random.Random(seed)
        pts = set()
        centers = [(rng.randint(-20, 20), rng.randint(-20, 20)) for _ in range(5)]
        while len(pts) < 30:
            cx, cy = rng.choice(centers)
            x = cx + rng.randint(-5, 5)
            y = cy + rng.randint(-5, 5)
            x = max(-30, min(30, x))
            y = max(-30, min(30, y))
            pts.add((x, y))
        return np.array(list(pts), dtype=np.int32)
    
    def make_random(seed=0):
        """Create random points."""
        rng = random.Random(seed)
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            pts.add((x, y))
        return np.array(list(pts), dtype=np.int32)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Standard grids
    initial_configs.append(make_grid(6, 5, 2, -5, -4))
    initial_configs.append(make_grid(5, 6, 2, -4, -5))
    initial_configs.append(make_grid(6, 5, 3, -7, -6))
    initial_configs.append(make_grid(5, 6, 3, -6, -7))
    
    # Sheared grids
    initial_configs.append(make_sheared_grid(6, 5, 2, 1, -5, -4))
    initial_configs.append(make_sheared_grid(5, 6, 2, 1, -4, -5))
    initial_configs.append(make_sheared_grid(6, 5, 2, -1, -5, -4))
    initial_configs.append(make_sheared_grid(5, 6, 2, -1, -4, -5))
    
    # Clustered random
    for seed in [42, 123, 456]:
        initial_configs.append(make_clustered_random(seed))
    
    # Random
    for seed in [0, 1, 2]:
        initial_configs.append(make_random(seed))
    
    # Evaluate all initial configs
    best_score = 0
    best_points = None
    best_area = 0
    
    for config in initial_configs:
        if len(config) != 30:
            continue
        if len(set(map(tuple, config.tolist()))) != 30:
            continue
        areas = compute_all_areas(config)
        score, area = get_max_multiplicity(areas)
        if score > best_score:
            best_score = score
            best_points = config.copy()
            best_area = area
    
    if best_points is None:
        best_points = make_random(0)
        areas = compute_all_areas(best_points)
        best_score, best_area = get_max_multiplicity(areas)
    
    # Phase 1: Greedy coordinate descent with exhaustive position search
    def greedy_coordinate_descent(points, target_area, time_limit):
        """
        For each point, exhaustively try all valid positions to maximize
        the count of triangles with doubled area == target_area.
        """
        pts = points.copy()
        occupied = set(map(tuple, pts.tolist()))
        areas = compute_all_areas(pts)
        curr_count = count_target(areas, target_area)
        best_count = curr_count
        best_pts = pts.copy()
        
        # Precompute which triples involve each point
        triples_by_point = [[] for _ in range(30)]
        for t in range(N_TRIPLES):
            for p in [TRI_I[t], TRI_J[t], TRI_K[t]]:
                triples_by_point[p].append(t)
        
        # Base areas: for each point i, areas of triples NOT involving i
        # We'll compute this on the fly for efficiency
        
        all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        
        improved = True
        passes = 0
        max_passes = 20
        
        while improved and passes < max_passes and time.time() < time_limit - 3:
            improved = False
            passes += 1
            
            for i in range(30):
                if time.time() > time_limit - 3:
                    break
                
                old_pt = (int(pts[i, 0]), int(pts[i, 1]))
                
                # Compute base areas: all triples not involving i
                # Triples involving i
                inv_triples = set(triples_by_point[i])
                
                # Base counter: areas of triples NOT involving i
                base_areas = np.zeros(N_TRIPLES, dtype=np.int32)
                for t in range(N_TRIPLES):
                    if t not in inv_triples:
                        pi = pts[TRI_I[t]]
                        pj = pts[TRI_J[t]]
                        pk = pts[TRI_K[t]]
                        det = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                        base_areas[t] = abs(det)
                
                base_nonzero = base_areas[base_areas > 0]
                base_counter = Counter(base_nonzero.tolist())
                base_max = max(base_counter.values()) if base_counter else 0
                
                best_new_pt = None
                best_new_count = curr_count
                
                # Try all valid positions
                for (x, y) in all_positions:
                    if (x, y) == old_pt:
                        continue
                    if (x, y) in occupied:
                        continue
                    
                    # Compute areas of triples involving i at new position
                    new_areas_list = []
                    for t in inv_triples:
                        a, b, c = TRI_I[t], TRI_J[t], TRI_K[t]
                        # One of a,b,c is i
                        if a == i:
                            p1 = (x, y)
                            p2 = tuple(pts[b])
                            p3 = tuple(pts[c])
                        elif b == i:
                            p1 = tuple(pts[a])
                            p2 = (x, y)
                            p3 = tuple(pts[c])
                        else:
                            p1 = tuple(pts[a])
                            p2 = tuple(pts[b])
                            p3 = (x, y)
                        
                        det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                        area = abs(det)
                        if area > 0:
                            new_areas_list.append(area)
                    
                    # Merge with base
                    merged_count_target = base_counter.get(target_area, 0)
                    for a in new_areas_list:
                        if a == target_area:
                            merged_count_target += 1
                    
                    if merged_count_target > best_new_count:
                        best_new_count = merged_count_target
                        best_new_pt = (x, y)
                
                if best_new_pt is not None:
                    occupied.remove(old_pt)
                    pts[i] = [best_new_pt[0], best_new_pt[1]]
                    occupied.add(best_new_pt)
                    curr_count = best_new_count
                    improved = True
                    if curr_count > best_count:
                        best_count = curr_count
                        best_pts = pts.copy()
        
        return best_count, best_pts
    
    # Phase 2: Multi-target optimization
    # First, find the best area from current best
    areas = compute_all_areas(best_points)
    best_score, best_area = get_max_multiplicity(areas)
    best_area = int(best_area)
    
    # Try greedy coordinate descent on the best area
    phase1_deadline = start_time + 30
    if time.time() < phase1_deadline:
        score, points = greedy_coordinate_descent(best_points, best_area, phase1_deadline)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Re-evaluate to get potentially different best area
    areas = compute_all_areas(best_points)
    new_score, new_area = get_max_multiplicity(areas)
    if new_score > best_score:
        best_score = new_score
        best_area = int(new_area)
    
    # Phase 3: Try a few other target areas
    phase3_deadline = deadline - 5
    if time.time() < phase3_deadline:
        # Get top 5 most frequent areas
        non_zero = areas[areas > 0]
        if len(non_zero) > 0:
            unique_areas, counts = np.unique(non_zero, return_counts=True)
            top_indices = np.argsort(counts)[::-1][:5]
            target_areas = [int(unique_areas[i]) for i in top_indices]
            
            for ta in target_areas:
                if ta == best_area:
                    continue
                if time.time() > phase3_deadline:
                    break
                remaining = phase3_deadline - time.time()
                if remaining < 5:
                    break
                score, points = greedy_coordinate_descent(best_points, ta, time.time() + min(remaining, 15))
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_area = ta
    
    # Phase 4: Final polish - one more pass on the best area
    if time.time() < deadline - 2:
        score, points = greedy_coordinate_descent(best_points, best_area, deadline - 1)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Final validation
    result = [(int(x), int(y)) for x, y in best_points.tolist()]
    assert len(result) == 30
    assert len(set(result)) == 30
    for x, y in result:
        assert -30 <= x <= 30
        assert -30 <= y <= 30
    
    return result
