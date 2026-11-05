"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container using a multi-strategy approach.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    bin=0 means loaded at (x,y,z) with dimensions (w,h,d).
    bin=-1 means left out (dimensions are the original box dims).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin out of 120s
    
    # Preprocess boxes: for each box, consider all 6 rotations
    # Each rotation is (w, h, d) where these are the dimensions along x, y, z
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    volumes = [w*h*d for w, h, d in boxes]
    
    def _check_time():
        return time.time() < deadline
    
    def _place_single_strategy(order, rotation_choice, max_iter=50000):
        """
        Place boxes in given order with given rotation choice strategy.
        rotation_choice: 0=original, 1=random best fit, 2=try all
        """
        if not _check_time():
            return None
        
        # Track occupied space using a grid or list of placed boxes
        placed_boxes = []  # list of (x, y, z, w, h, d)
        placement_result = [None] * n
        
        # For efficiency, we'll use a simple list-based overlap check
        # For 60 boxes this is O(n^2) which is fine
        
        def _overlaps(x, y, z, w, h, d):
            for px, py, pz, pw, ph, pd in placed_boxes:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return True
            return False
        
        def _fits(x, y, z, w, h, d):
            return (x >= 0 and y >= 0 and z >= 0 and
                    x + w <= W and y + h <= H and z + d <= D)
        
        def _find_position(w, h, d):
            """Find a valid position for a box with given dimensions using corner-based heuristic."""
            # Try to place at the "lowest" possible position
            # First, try to place against existing boxes or walls
            
            # Generate candidate positions
            candidates = []
            
            # Candidate: origin
            candidates.append((0, 0, 0))
            
            # Candidates: adjacent to each placed box
            for px, py, pz, pw, ph, pd in placed_boxes:
                # Place to the right
                candidates.append((px + pw, py, pz))
                # Place above
                candidates.append((px, py + ph, pz))
                # Place in front
                candidates.append((px, py, pz + pd))
                # Place at the corner combinations
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
                # Also try aligning with the faces
                candidates.append((px, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py, pz + pd))
            
            # Deduplicate and sort by "compactness" (prefer lower positions)
            seen = set()
            unique_candidates = []
            for cx, cy, cz in candidates:
                if (cx, cy, cz) not in seen:
                    seen.add((cx, cy, cz))
                    unique_candidates.append((cx, cy, cz))
            
            # Sort by z, then y, then x to prefer lower positions
            unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for cx, cy, cz in unique_candidates:
                if _fits(cx, cy, cz, w, h, d) and not _overlaps(cx, cy, cz, w, h, d):
                    return (cx, cy, cz)
            
            return None
        
        for idx in order:
            if not _check_time():
                return None
            
            orig_w, orig_h, orig_d = boxes[idx]
            
            if rotation_choice == 0:
                # Use original orientation
                w, h, d = orig_w, orig_h, orig_d
                pos = _find_position(w, h, d)
                if pos is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                    continue
                else:
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
            elif rotation_choice == 1:
                # Try all rotations, pick the one that fits
                best_rot = None
                for rot in box_rotations[idx]:
                    w, h, d = rot
                    pos = _find_position(w, h, d)
                    if pos is not None:
                        best_rot = (w, h, d, pos)
                        break
                if best_rot is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                else:
                    w, h, d, pos = best_rot
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
            else:
                # Try all rotations, pick the one that minimizes the "corner distance"
                best_rot = None
                best_score = float('inf')
                for rot in box_rotations[idx]:
                    w, h, d = rot
                    pos = _find_position(w, h, d)
                    if pos is not None:
                        # Score: prefer positions closer to origin
                        score = pos[0] + pos[1] + pos[2]
                        if score < best_score:
                            best_score = score
                            best_rot = (w, h, d, pos)
                if best_rot is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                else:
                    w, h, d, pos = best_rot
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
        
        # Ensure all placements are set
        for i in range(n):
            if placement_result[i] is None:
                orig_w, orig_h, orig_d = boxes[i]
                placement_result[i] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
        
        return placement_result
    
    # Strategy 1: Sort by volume descending, try all rotations
    order1 = sorted(box_indices, key=lambda i: volumes[i], reverse=True)
    result1 = _place_single_strategy(order1, 2)
    
    if result1 is not None:
        # Calculate score
        filled_vol = sum(w*h*d for bin_id, x, y, z, w, h, d in result1 if bin_id == 0)
        container_vol = W * H * D
        score1 = filled_vol / container_vol if container_vol > 0 else 0
    else:
        score1 = 0
    
    # Strategy 2: Sort by max dimension descending
    order2 = sorted(box_indices, key=lambda i: max(boxes[i]), reverse=True)
    if _check_time():
        result2 = _place_single_strategy(order2, 2)
        if result2 is not None:
            filled_vol = sum(w*h*d for bin_id, x, y, z, w, h, d in result2 if bin_id == 0)
            container_vol = W * H * D
            score2 = filled_vol / container_vol if container_vol > 0 else 0
        else:
            score2 = 0
    else:
        result2 = None
        score2 = 0
    
    # Strategy 3: Sort by minimum dimension ascending (fit small gaps)
    order3 = sorted(box_indices, key=lambda i: min(boxes[i]))
    if _check_time():
        result3 = _place_single_strategy(order3, 2)
        if result3 is not None:
            filled_vol = sum(w*h*d for bin_id, x, y, z, w, h, d in result3 if bin_id == 0)
            container_vol = W * H * D
            score3 = filled_vol / container_vol if container_vol > 0 else 0
        else:
            score3 = 0
    else:
        result3 = None
        score3 = 0
    
    # Strategy 4: Random order with multiple restarts
    best_result = None
    best_score = max(score1, score2, score3)
    
    if result1 is not None and score1 == best_score:
        best_result = result1
    if result2 is not None and score2 == best_score:
        best_result = result2
    if result3 is not None and score3 == best_score:
        best_result = result3
    
    # Try random restarts if we have time
    rng = random.Random(seed)
    max_restarts = 20
    for r in range(max_restarts):
        if not _check_time():
            break
        order_rand = box_indices[:]
        rng.shuffle(order_rand)
        result_r = _place_single_strategy(order_rand, 2)
        if result_r is not None:
            filled_vol = sum(w*h*d for bin_id, x, y, z, w, h, d in result_r if bin_id == 0)
            container_vol = W * H * D
            score_r = filled_vol / container_vol if container_vol > 0 else 0
            if score_r > best_score:
                best_score = score_r
                best_result = result_r
    
    if best_result is None:
        # Fallback: return all as left out
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return best_result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Try the main algorithm
    try:
        result = _try_place(boxes, container, seed=42)
        return result
    except Exception:
        # Fallback to simple greedy
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
# EVOLVE-BLOCK-END
