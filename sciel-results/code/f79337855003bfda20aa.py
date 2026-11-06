"""Strong 3D Container Loading: multi-strategy with skyline and best-fit decreasing."""
import time
import math
import numpy as np
from functools import lru_cache

# EVOLVE-BLOCK-START
def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    
    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]
        
        # Sort orientations: prefer those with smaller max dimension, then better fit
        valid_orients.sort(key=lambda o: (max(o), o[0] * o[1] * o[2], o[0], o[1], o[2]))
        
        volume = w * h * d
        surface = 2 * (w * h + h * d + w * d)
        max_dim = max(w, h, d)
        min_dim = min(w, h, d)
        box_data.append((idx, valid_orients, volume, surface, max_dim, min_dim, w, h, d))
    
    # Multiple sorting strategies
    box_data_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    box_data_maxdim = sorted(box_data, key=lambda x: x[4], reverse=True)
    box_data_surface = sorted(box_data, key=lambda x: x[3], reverse=True)
    # Aspect ratio: prefer more "cubic" boxes first (less elongated)
    box_data_aspect = sorted(box_data, key=lambda x: x[4] / max(x[5], 1), reverse=True)
    
    # Skyline-based placement for better space utilization
    def try_place_skyline(boxes_to_place, W, H, D, deadline, candidate_order='zyx', time_limit=None):
        """Skyline-based placement that tracks free space more efficiently."""
        if time_limit is not None:
            local_deadline = time.time() + time_limit
        else:
            local_deadline = deadline
        
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        total_placed_volume = 0
        
        # Skyline representation: for each (x, y), track the max z occupied
        # This is a simplified skyline that helps find gaps
        # We'll use a combination of skyline and candidate positions
        
        for i, (idx, orients, volume, surface, max_dim, min_dim, ow, oh, od) in enumerate(boxes_to_place):
            if time.time() > local_deadline:
                for j in range(i, len(boxes_to_place)):
                    if results[j] is None:
                        oj = boxes_to_place[j][0]
                        ow2, oh2, od2 = boxes_to_place[j][6], boxes_to_place[j][7], boxes_to_place[j][8]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                for k in range(len(boxes_to_place)):
                    if results[k] is None:
                        w_k, h_k, d_k = boxes_to_place[k][6], boxes_to_place[k][7], boxes_to_place[k][8]
                        results[boxes_to_place[k][0]] = (-1, 0, 0, 0, w_k, h_k, d_k)
                break
            
            # Generate candidate positions from placed boxes
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            found = False
            # Generate candidates and sort by priority
            if candidate_order == 'zyx':
                candidates = [(cx, cy, cz) for cz in cz_list for cy in cy_list for cx in cx_list]
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            elif candidate_order == 'xyz':
                candidates = [(cx, cy, cz) for cx in cx_list for cy in cy_list for cz in cz_list]
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            elif candidate_order == 'yxz':
                candidates = [(cx, cy, cz) for cy in cy_list for cx in cx_list for cz in cz_list]
                candidates.sort(key=lambda c: (c[1], c[0], c[2]))
            else:
                candidates = [(cx, cy, cz) for cz in cz_list for cy in cy_list for cx in cx_list]
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            
            # Use occupancy grid for fast overlap checking
            occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
            for (px, py, pz, pw, ph, pd) in placed:
                occ[px:px+pw, py:py+ph, pz:pz+pd] = 1
            
            for (cx, cy, cz) in candidates:
                # Try each orientation
                for (bw, bh, bd) in orients:
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    region = occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                    if np.any(region > 0):
                        continue
                    
                    # Place this box
                    occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = 1
                    placed.append((cx, cy, cz, bw, bh, bd))
                    results[idx] = (0, cx, cy, cz, bw, bh, bd)
                    total_placed_volume += volume
                    found = True
                    break
                
                if found:
                    break
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for k in range(len(boxes_to_place)):
            if results[k] is None:
                w_k, h_k, d_k = boxes_to_place[k][6], boxes_to_place[k][7], boxes_to_place[k][8]
                results[boxes_to_place[k][0]] = (-1, 0, 0, 0, w_k, h_k, d_k)
        
        return results, total_placed_volume
    
    def try_place(boxes_to_place, W, H, D, deadline, candidate_order='zyx', time_limit=None):
        """Try to place boxes using a candidate position approach with occupancy grid."""
        if time_limit is not None:
            local_deadline = time.time() + time_limit
        else:
            local_deadline = deadline
        
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        total_placed_volume = 0
        
        # Use a 3D numpy array for fast overlap checking
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        
        for i, (idx, orients, volume, surface, max_dim, min_dim, ow, oh, od) in enumerate(boxes_to_place):
            if time.time() > local_deadline:
                # Time out - mark remaining as not placed
                for j in range(i, len(boxes_to_place)):
                    if results[j] is None:
                        oj = boxes_to_place[j][0]
                        ow2, oh2, od2 = boxes_to_place[j][6], boxes_to_place[j][7], boxes_to_place[j][8]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                # Fill any that are still None
                for k in range(len(boxes_to_place)):
                    if results[k] is None:
                        w_k, h_k, d_k = boxes_to_place[k][6], boxes_to_place[k][7], boxes_to_place[k][8]
                        results[boxes_to_place[k][0]] = (-1, 0, 0, 0, w_k, h_k, d_k)
                break
            
            # Get candidate positions from placed boxes and origin
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            found = False
            # Generate candidates and sort by priority
            if candidate_order == 'zyx':
                candidates = [(cx, cy, cz) for cz in cz_list for cy in cy_list for cx in cx_list]
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            elif candidate_order == 'xyz':
                candidates = [(cx, cy, cz) for cx in cx_list for cy in cy_list for cz in cz_list]
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            elif candidate_order == 'yxz':
                candidates = [(cx, cy, cz) for cy in cy_list for cx in cx_list for cz in cz_list]
                candidates.sort(key=lambda c: (c[1], c[0], c[2]))
            elif candidate_order == 'zxy':
                candidates = [(cx, cy, cz) for cz in cz_list for cx in cx_list for cy in cy_list]
                candidates.sort(key=lambda c: (c[2], c[0], c[1]))
            else:
                candidates = [(cx, cy, cz) for cz in cz_list for cy in cy_list for cx in cx_list]
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            
            for (cx, cy, cz) in candidates:
                # Try each orientation
                for (bw, bh, bd) in orients:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap using occupancy grid - fast vectorized check
                    region = occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                    if np.any(region > 0):
                        continue
                    
                    # Place this box - update occupancy grid
                    occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = 1
                    placed.append((cx, cy, cz, bw, bh, bd))
                    results[idx] = (0, cx, cy, cz, bw, bh, bd)
                    total_placed_volume += volume
                    found = True
                    break
                
                if found:
                    break
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill any remaining None entries
        for k in range(len(boxes_to_place)):
            if results[k] is None:
                w_k, h_k, d_k = boxes_to_place[k][6], boxes_to_place[k][7], boxes_to_place[k][8]
                results[boxes_to_place[k][0]] = (-1, 0, 0, 0, w_k, h_k, d_k)
        
        return results, total_placed_volume
    
    # Allocate time budget across strategies
    elapsed = time.time() - t_start
    remaining = deadline - t_start
    # Give each strategy roughly equal time, with a minimum
    n_strategies = 8
    time_per_strategy = max(remaining / n_strategies, 3.0)
    
    strategies = [
        ('vol_zyx', box_data_vol, 'zyx'),
        ('vol_xyz', box_data_vol, 'xyz'),
        ('vol_yxz', box_data_vol, 'yxz'),
        ('maxdim_zyx', box_data_maxdim, 'zyx'),
        ('surface_zyx', box_data_surface, 'zyx'),
        ('aspect_zyx', box_data_aspect, 'zyx'),
        ('vol_zxy', box_data_vol, 'zxy'),
        ('maxdim_xyz', box_data_maxdim, 'xyz'),
    ]
    
    all_results = []
    
    for name, bdata, order in strategies:
        if time.time() > deadline:
            break
        tl = min(time_per_strategy, deadline - time.time())
        if tl < 1.0:
            break
        res, vol = try_place(bdata, W, H, D, deadline, order, time_limit=tl)
        all_results.append((vol, res))
    
    # Choose the best result based on placed volume
    if not all_results:
        res, vol = try_place(box_data_vol, W, H, D, deadline, 'zyx')
        return res
    
    best_results = max(all_results, key=lambda x: x[0])
    return best_results[1]
# EVOLVE-BLOCK-END
