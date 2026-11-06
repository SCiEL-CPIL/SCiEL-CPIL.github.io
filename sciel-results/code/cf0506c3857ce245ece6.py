"""Strong 3D Container Loading: multi-strategy with skyline and best-fit decreasing."""
import time
from itertools import product

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
    box_data.sort(key=lambda x: x[2], reverse=True)
    
    def try_place(boxes_to_place, W, H, D, deadline):
        """Try to place boxes using a candidate position approach."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        # Use a set of candidate corner points
        # Start with (0,0,0)
        
        def get_candidates():
            """Generate candidate positions from placed boxes and origin."""
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
            return cx_list, cy_list, cz_list
        
        total_placed_volume = 0
        
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
            
            # Get candidate positions
            cx_list, cy_list, cz_list = get_candidates()
            
            found = False
            # Try each candidate position, prefer lower corners (bottom-left-front)
            # Sort candidates by (z, y, x) to prefer lowest positions
            candidates = []
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        candidates.append((cz, cy, cx))
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            
            for (cz, cy, cx) in candidates:
                # Try each orientation
                for (bw, bh, bd) in orients:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        # No overlap if separated in any dimension
                        if (cx + bw <= px) or (px + pw <= cx) or \
                           (cy + bh <= py) or (py + ph <= cy) or \
                           (cz + bd <= pz) or (pz + pd <= cz):
                            continue
                        # Overlap
                        overlap = True
                        break
                    
                    if not overlap:
                        # Place this box
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
    
    # Strategy 1: Standard volume-descending order
    results1, vol1 = try_place(box_data, W, H, D, deadline)
    
    # Strategy 2: Try with different sorting - maybe by max dimension
    box_data2 = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]), reverse=True)
    results2, vol2 = try_place(box_data2, W, H, D, deadline)
    
    # Strategy 3: Try with different candidate ordering (x, y, z instead of z, y, x)
    def try_place_xyz(boxes_to_place, W, H, D, deadline):
        """Try to place boxes using a candidate position approach with XYZ ordering."""
        placed = []
        results = [None] * len(boxes_to_place)
        
        def get_candidates():
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
            return cx_list, cy_list, cz_list
        
        total_placed_volume = 0
        
        for idx, orients, volume, ow, oh, od in boxes_to_place:
            if time.time() > deadline:
                for j in range(idx, len(boxes_to_place)):
                    if results[j] is None:
                        oj, oorient, ov, ow2, oh2, od2 = boxes_to_place[j]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                for i in range(len(boxes_to_place)):
                    if results[i] is None:
                        w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                        results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                break
            
            cx_list, cy_list, cz_list = get_candidates()
            
            found = False
            candidates = []
            for cx in cx_list:
                for cy in cy_list:
                    for cz in cz_list:
                        candidates.append((cx, cy, cz))
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            
            for (cx, cy, cz) in candidates:
                for (bw, bh, bd) in orients:
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if (cx + bw <= px) or (px + pw <= cx) or \
                           (cy + bh <= py) or (py + ph <= cy) or \
                           (cz + bd <= pz) or (pz + pd <= cz):
                            continue
                        overlap = True
                        break
                    
                    if not overlap:
                        placed.append((cx, cy, cz, bw, bh, bd))
                        results[idx] = (0, cx, cy, cz, bw, bh, bd)
                        total_placed_volume += volume
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(len(boxes_to_place)):
            if results[i] is None:
                w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        return results, total_placed_volume
    
    results3, vol3 = try_place_xyz(box_data, W, H, D, deadline)
    
    # Choose the best result based on placed volume
    best_results = max([(vol1, results1), (vol2, results2), (vol3, results3)], key=lambda x: x[0])
    
    return best_results[1]
# EVOLVE-BLOCK-END
