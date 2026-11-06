"""Strong 3D Container Loading: multi-strategy with skyline and best-fit decreasing."""
import time
import math
import numpy as np

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
        
        # Sort orientations: prefer those with smaller max dimension (more "flat")
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort boxes by volume descending (largest first)
    box_data_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    box_data_maxdim = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]), reverse=True)
    
    def try_place(boxes_to_place, W, H, D, deadline, candidate_order='zyx'):
        """Try to place boxes using a candidate position approach with occupancy grid."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        total_placed_volume = 0
        
        # Use a 3D numpy array for fast overlap checking
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        
        for idx, orients, volume, ow, oh, od in boxes_to_place:
            if time.time() > deadline:
                # Time out - mark remaining as not placed
                for j in range(idx, len(boxes_to_place)):
                    if results[j] is None:
                        oj, oorient, ov, ow2, oh2, od2 = boxes_to_place[j]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                # Fill any that are still None
                for i in range(len(boxes_to_place)):
                    if results[i] is None:
                        w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                        results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
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
            candidates = []
            if candidate_order == 'zyx':
                for cz in cz_list:
                    for cy in cy_list:
                        for cx in cx_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))  # z, y, x
            elif candidate_order == 'xyz':
                for cx in cx_list:
                    for cy in cy_list:
                        for cz in cz_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))  # x, y, z
            elif candidate_order == 'yxz':
                for cy in cy_list:
                    for yx in cx_list:
                        for cz in cz_list:
                            candidates.append((yx, cy, cz))
                candidates.sort(key=lambda c: (c[1], c[0], c[2]))  # y, x, z
            elif candidate_order == 'zxy':
                for cz in cz_list:
                    for cx in cx_list:
                        for cy in cy_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[2], c[0], c[1]))  # z, x, y
            else:
                # default zyx
                for cz in cz_list:
                    for cy in cy_list:
                        for cx in cx_list:
                            candidates.append((cx, cy, cz))
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
        for i in range(len(boxes_to_place)):
            if results[i] is None:
                w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        return results, total_placed_volume
    
    # Strategy 1: Volume-descending with ZYX candidate ordering
    results1, vol1 = try_place(box_data_vol, W, H, D, deadline, 'zyx')
    
    # Strategy 2: Max-dimension-descending with ZYX
    results2, vol2 = try_place(box_data_maxdim, W, H, D, deadline, 'zyx')
    
    # Strategy 3: Volume-descending with XYZ candidate ordering
    results3, vol3 = try_place(box_data_vol, W, H, D, deadline, 'xyz')
    
    # Strategy 4: Volume-descending with YXZ candidate ordering
    results4, vol4 = try_place(box_data_vol, W, H, D, deadline, 'yxz')
    
    # Strategy 5: Volume-descending with ZXY candidate ordering
    results5, vol5 = try_place(box_data_vol, W, H, D, deadline, 'zxy')
    
    # Choose the best result based on placed volume
    all_results = [(vol1, results1), (vol2, results2), (vol3, results3), (vol4, results4), (vol5, results5)]
    best_results = max(all_results, key=lambda x: x[0])
    
    return best_results[1]
# EVOLVE-BLOCK-END
