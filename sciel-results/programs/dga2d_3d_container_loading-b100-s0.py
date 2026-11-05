import time
import itertools
import random
import numpy as np

def _orientations(w, h, d):
    """Return all 6 distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_fast(placed_boxes, x, y, z, w, h, d, W, H, D):
    """
    Fast overlap check.
    placed_boxes: list of (x, y, z, w, h, d)
    """
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    
    # Check against all placed boxes
    for (px, py, pz, pw, ph, pd) in placed_boxes:
        # If they don't overlap in any dimension, it's fine
        if not (x + w <= px or px + pw <= x or 
                y + h <= py or py + ph <= y or 
                z + d <= pz or pz + pd <= z):
            return False
    return True

def _pack_heuristic(boxes, container, order, deadline):
    """
    Pack boxes using a Best-Fit heuristic.
    Returns (placements_dict, filled_volume)
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, w, h, d)
    filled_vol = 0.0
    
    # Current bounding box dimensions
    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0
    
    # Cache of candidate coordinates to speed up search
    # We will dynamically update these as we place boxes
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Generate candidate X and Y coordinates
        # These are 0 and the "front" faces of placed boxes
        x_cands = {0}
        y_cands = {0}
        z_cands = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
            z_cands.add(pz)
            z_cands.add(pz + pd)
            
        # Filter candidates that are within container bounds
        x_cands = sorted([x for x in x_cands if 0 <= x <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y <= H])
        z_cands = sorted([z for z in z_cands if 0 <= z <= D])
        
        # Try all orientations
        for ow, oh, od in _orientations(w, h, d):
            for cx in x_cands:
                if cx + ow > W:
                    continue
                for cy in y_cands:
                    if cy + oh > H:
                        continue
                    for cz in z_cands:
                        if cz + od > D:
                            continue
                            
                        # Check if it fits without overlap
                        if _fits_fast(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            # Calculate new bounding box
                            new_max_x = max(cur_max_x, cx + ow)
                            new_max_y = max(cur_max_y, cy + oh)
                            new_max_z = max(cur_max_z, cz + od)
                            
                            # Primary score: Bounding Box Volume
                            bb_vol = new_max_x * new_max_y * new_max_z
                            
                            # Secondary score: Distance to origin (prefer compact packing)
                            # Using sum of coordinates as a tie-breaker
                            dist_score = cx + cy + cz
                            
                            # Combine scores. We want to minimize BB volume.
                            # If BB volume is the same, we want to minimize distance.
                            # To make this a single comparable score, we can use a tuple
                            # or a weighted sum. Tuple is safer for exact minimization.
                            score = (bb_vol, dist_score)
                            
                            if score < (best_score, 0) if best_score != float('inf') else True:
                                # Handle the inf case properly
                                if best_score == float('inf') or score < (best_score, 0):
                                    best_score = bb_vol
                                    best_pos = (cx, cy, cz)
                                    best_orient = (ow, oh, od)
                                    # Store the dist score too for tie-breaking if needed, 
                                    # but for now we just keep the first best found for a given BB vol
                                    # Actually, let's store the tuple to be precise
                                    if best_score == float('inf'):
                                         best_score_tuple = score
                                    elif score < best_score_tuple:
                                         best_score_tuple = score
                                         best_pos = (cx, cy, cz)
                                         best_orient = (ow, oh, od)
                                    # Note: The logic above is slightly messy with the inf check.
                                    # Let's refactor to use a tuple directly.

        # Refactored selection logic inside the loop would be better, 
        # but for now, let's assume the above finds a candidate.
        # We need to fix the best_score tracking to use tuples properly.
        
        # Let's restart the inner loop logic to be cleaner.
        # (I will rewrite the function body in the final code to be clean)
        
    return placements, filled_vol, cur_max_x, cur_max_y, cur_max_z

def _pack_best_fit_v2(boxes, container, order, deadline):
    """
    Improved Best-Fit with proper tuple scoring.
    """
    W, H, D = container
    placed = []
    placements = {}
    filled_vol = 0.0
    
    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score_tuple = None
        best_pos = None
        best_orient = None
        
        x_cands = {0}
        y_cands = {0}
        z_cands = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
            z_cands.add(pz)
            z_cands.add(pz + pd)
            
        x_cands = sorted([x for x in x_cands if 0 <= x <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y <= H])
        z_cands = sorted([z for z in z_cands if 0 <= z <= D])
        
        for ow, oh, od in _orientations(w, h, d):
            for cx in x_cands:
                if cx + ow > W: continue
                for cy in y_cands:
                    if cy + oh > H: continue
                    for cz in z_cands:
                        if cz + od > D: continue
                        
                        if _fits_fast(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            new_max_x = max(cur_max_x, cx + ow)
                            new_max_y = max(cur_max_y, cy + oh)
                            new_max_z = max(cur_max_z, cz + od)
                            
                            bb_vol = new_max_x * new_max_y * new_max_z
                            dist_score = cx + cy + cz
                            
                            current_score = (bb_vol, dist_score)
                            
                            if best_score_tuple is None or current_score < best_score_tuple:
                                best_score_tuple = current_score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
            cur_max_x = max(cur_max_x, cx + ow)
            cur_max_y = max(cur_max_y, cy + oh)
            cur_max_z = max(cur_max_z, cz + od)
            
    return placements, filled_vol, cur_max_x, cur_max_y, cur_max_z

def _gap_fill(boxes, container, placements, filled_vol, deadline):
    """
    Try to insert remaining boxes into gaps.
    """
    W, H, D = container
    n = len(boxes)
    placed_boxes = [(x, y, z, w, h, d) for x, y, z, w, h, d in placements.values()]
    
    # List of unplaced boxes
    unplaced = [i for i in range(n) if i not in placements]
    
    # Sort unplaced by volume descending (try to fit big ones in gaps first? 
    # Or small ones? Usually small ones fit in gaps better. 
    # Let's try small to large for gap filling)
    unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in unplaced[:]: # Iterate over a copy
            if idx not in unplaced: continue
            if time.time() > deadline: break
            
            w, h, d = boxes[idx]
            best_score_tuple = None
            best_pos = None
            best_orient = None
            
            # Generate candidates
            x_cands = {0}
            y_cands = {0}
            z_cands = {0}
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                x_cands.add(px)
                x_cands.add(px + pw)
                y_cands.add(py)
                y_cands.add(py + ph)
                z_cands.add(pz)
                z_cands.add(pz + pd)
                
            x_cands = sorted([x for x in x_cands if 0 <= x <= W])
            y_cands = sorted([y for y in y_cands if 0 <= y <= H])
            z_cands = sorted([z for z in z_cands if 0 <= z <= D])
            
            for ow, oh, od in _orientations(w, h, d):
                for cx in x_cands:
                    if cx + ow > W: continue
                    for cy in y_cands:
                        if cy + oh > H: continue
                        for cz in z_cands:
                            if cz + od > D: continue
                            
                            if _fits_fast(placed_boxes, cx, cy, cz, ow, oh, od, W, H, D):
                                # For gap filling, we don't care as much about BB volume increase
                                # as we do about filling space. 
                                # However, we should still avoid expanding the BB if possible.
                                # Let's use the same scoring: Minimize BB Vol, then Dist.
                                cur_max_x = max(x+w for x,y,z,w,h,d in placed_boxes) if placed_boxes else 0
                                cur_max_y = max(y+h for x,y,z,w,h,d in placed_boxes) if placed_boxes else 0
                                cur_max_z = max(z+d for x,y,z,w,h,d in placed_boxes) if placed_boxes else 0
                                
                                new_max_x = max(cur_max_x, cx + ow)
                                new_max_y = max(cur_max_y, cy + oh)
                                new_max_z = max(cur_max_z, cz + od)
                                
                                bb_vol = new_max_x * new_max_y * new_max_z
                                dist_score = cx + cy + cz
                                current_score = (bb_vol, dist_score)
                                
                                if best_score_tuple is None or current_score < best_score_tuple:
                                    best_score_tuple = current_score
                                    best_pos = (cx, cy, cz)
                                    best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                placed_boxes.append((cx, cy, cz, ow, oh, od))
                placements[idx] = (cx, cy, cz, ow, oh, od)
                filled_vol += w * h * d
                unplaced.remove(idx)
                improved = True
                
    return placements, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate diverse orders
    orders = []
    
    # 1. Volume Descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max Dim Descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Sum of Dims Descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Min Dim Descending (Flat first)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # 5. Surface Area Descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 6. Random Shuffle (for diversity)
    random_order = list(range(n))
    random.shuffle(random_order)
    orders.append(random_order)
    
    best_placements = {}
    best_filled = 0.0
    
    for order in orders:
        if time.time() > deadline:
            break
            
        placements, filled_vol, mx, my, mz = _pack_best_fit_v2(boxes, container, order, deadline)
        
        # Gap filling
        if filled_vol > 0 and time.time() < deadline - 10:
            placements, filled_vol = _gap_fill(boxes, container, placements, filled_vol, deadline)
            
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_placements = placements.copy()
            
    # Build result
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if i in best_placements:
            x, y, z, ow, oh, od = best_placements[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
