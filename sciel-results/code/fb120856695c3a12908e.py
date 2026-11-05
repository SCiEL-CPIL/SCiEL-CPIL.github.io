import time
import numpy as np
import itertools

def get_rotations(w, h, d):
    """
    Returns a list of unique (w, h, d) tuples for all 6 rotations.
    """
    rotations = [
        (w, h, d),
        (w, d, h),
        (h, w, d),
        (h, d, w),
        (d, w, h),
        (d, h, w)
    ]
    # Remove duplicates
    seen = set()
    unique_rotations = []
    for r in rotations:
        if r not in seen:
            seen.add(r)
            unique_rotations.append(r)
    return unique_rotations

def try_pack(boxes, container, sort_key):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    # Create list of (box, original_index)
    indexed_boxes = [(boxes[i], i) for i in range(n)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0]))
    
    placed_boxes = [] # List of (x, y, z, w, h, d)
    candidates = set([(0, 0, 0)])
    results = [None] * n
    total_volume = 0
    
    # Store placed boxes in a numpy array for fast overlap checks
    # Shape: (N, 6) -> [x, y, z, w, h, d]
    placed_array = np.empty((0, 6), dtype=np.float64)
    
    for box, idx in indexed_boxes:
        bw, bh, bd = box
        best_pos = None
        best_rot = None
        best_score = None
        
        rotations = get_rotations(bw, bh, bd)
        
        # Sort candidates by z, then y, then x to prioritize bottom-left-back
        # This allows early termination if we find a very good spot? 
        # No, we must check all to find the global min score for this box.
        # But we can break early if we find (0,0,0) which is the absolute best.
        sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
        
        found = False
        for pos in sorted_candidates:
            px, py, pz = pos
            # If we already found a position with score (0,0,0), we can't do better
            if best_score == (0, 0, 0):
                break
                
            for rot in rotations:
                rw, rh, rd = rot
                
                # Check bounds
                if px + rw > W or py + rh > H or pz + rd > D:
                    continue
                
                # Check overlap using NumPy
                if len(placed_array) > 0:
                    # placed_array columns: 0=x, 1=y, 2=z, 3=w, 4=h, 5=d
                    # New box: [px, py, pz, rw, rh, rd]
                    # Overlap condition:
                    # px < p_x + p_w AND p_x < px + rw
                    # py < p_y + p_h AND p_y < py + rh
                    # pz < p_z + p_d AND p_z < pz + rd
                    
                    # Calculate overlaps for all placed boxes at once
                    # p_x, p_y, p_z are columns 0, 1, 2
                    # p_w, p_h, p_d are columns 3, 4, 5
                    
                    # px < p_x + p_w  =>  px - p_w < p_x
                    # p_x < px + rw   =>  p_x < px + rw
                    # py < p_y + p_h  =>  py - p_h < p_y
                    # p_y < py + rh   =>  p_y < py + rh
                    # pz < p_z + p_d  =>  pz - p_d < p_z
                    # p_z < pz + rd   =>  p_z < pz + rd
                    
                    # Let's construct the conditions
                    # p_x = placed_array[:, 0], p_w = placed_array[:, 3]
                    # Condition 1: px < p_x + p_w  =>  px - p_w < p_x
                    cond1 = px < placed_array[:, 0] + placed_array[:, 3]
                    cond2 = placed_array[:, 0] < px + rw
                    cond3 = py < placed_array[:, 1] + placed_array[:, 4]
                    cond4 = placed_array[:, 1] < py + rh
                    cond5 = pz < placed_array[:, 2] + placed_array[:, 5]
                    cond6 = placed_array[:, 2] < pz + rd
                    
                    overlap_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
                    if np.any(overlap_mask):
                        continue
                else:
                    # No placed boxes, so no overlap
                    pass
                
                if not (len(placed_array) > 0 and np.any(overlap_mask)):
                    score = (pz, py, px)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = rot
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw_r, bh_r, bd_r = best_rot
            # Place the box
            placed_boxes.append((bx, by, bz, bw_r, bh_r, bd_r))
            # Update numpy array
            new_box = np.array([[bx, by, bz, bw_r, bh_r, bd_r]], dtype=np.float64)
            if len(placed_array) == 0:
                placed_array = new_box
            else:
                placed_array = np.vstack((placed_array, new_box))
                
            total_volume += bw_r * bh_r * bd_r
            results[idx] = (0, bx, by, bz, bw_r, bh_r, bd_r)
            
            # Update candidates
            # Add corners of the placed box
            candidates.add((bx + bw_r, by, bz))
            candidates.add((bx, by + bh_r, bz))
            candidates.add((bx, by, bz + bd_r))
            candidates.add((bx + bw_r, by + bh_r, bz))
            candidates.add((bx + bw_r, by, bz + bd_r))
            candidates.add((bx, by + bh_r, bz + bd_r))
            candidates.add((bx + bw_r, by + bh_r, bz + bd_r))
        else:
            results[idx] = (-1, 0, 0, 0, bw, bh, bd)
            
    return results, total_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    n = len(boxes)
    if n == 0:
        return []
        
    # Define different sorting strategies
    sort_strategies = [
        lambda b: -(b[0] * b[1] * b[2]), # Volume descending
        lambda b: -max(b), # Max dimension descending
        lambda b: -(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]), # Surface area descending
        lambda b: -min(b), # Min dimension descending
        lambda b: -(b[0] + b[1] + b[2]), # Sum of dimensions descending
        lambda b: -abs(b[0] - b[1]), # Difference between first two dims (less negative is better? No, we want max volume, so maybe sort by volume then by aspect ratio?)
    ]
    
    best_results = None
    best_volume = -1
    
    for strategy in sort_strategies:
        if time.time() > deadline - 10: # Safety margin
            break
            
        results, vol = try_pack(boxes, container, strategy)
        if vol > best_volume:
            best_volume = vol
            best_results = results
            
    if best_results is None:
        # Fallback: return all left out
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
