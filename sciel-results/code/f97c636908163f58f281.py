"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return True
    return False

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container boundaries."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)

def _try_place_optimized(boxes, container, sort_key='volume', candidate_order='zyx'):
    """
    Attempt to place boxes into the container using an optimized candidate generation.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Filter rotations that fit in container
        valid_rots = [(rw, rh, rd) for rw, rh, rd in rots if rw <= W and rh <= H and rd <= D]
        if not valid_rots:
            valid_rots = [(w, h, d)]
        box_rotations.append(valid_rots)
    
    # Sort boxes based on the chosen key
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    if sort_key == 'volume':
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'max_dim':
        box_indices.sort(key=lambda i: max(boxes[i]), reverse=True)
    elif sort_key == 'surface_area':
        box_indices.sort(key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    elif sort_key == 'random':
        rng = random.Random(42)
        rng.shuffle(box_indices)
    else:
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)

    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    # Coordinate sets for candidate generation (Skyline approach)
    cx_set = {0}
    cy_set = {0}
    cz_set = {0}
    
    for idx in box_indices:
        if time.time() > deadline:
            break
            
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate lists from current extreme points
        cx_list = sorted(cx_set)
        cy_list = sorted(cy_set)
        cz_list = sorted(cz_set)
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Iterate candidates in specific order to prioritize "filling"
        # Order 'zyx': lowest z, then lowest y, then lowest x
        # This tends to fill the bottom layer first.
        
        if candidate_order == 'zyx':
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        for rw, rh, rd in rots:
                            if cx + rw > W or cy + rh > H or cz + rd > D:
                                continue
                            
                            # Quick check: does it overlap?
                            if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                                continue
                            
                            # Score: prefer lower z, then y, then x
                            # We can break early if we find a perfect fit at the lowest possible spot?
                            # Not necessarily, because a slightly higher z might allow better packing later?
                            # But for greedy, lowest is usually best.
                            score = cz * 10000 + cy * 100 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        elif candidate_order == 'xyz':
            for cx in cx_list:
                for cy in cy_list:
                    for cz in cz_list:
                        for rw, rh, rd in rots:
                            if cx + rw > W or cy + rh > H or cz + rd > D:
                                continue
                            if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                                continue
                            score = cx * 10000 + cy * 100 + cz
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        else: # default zyx
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        for rw, rh, rd in rots:
                            if cx + rw > W or cy + rh > H or cz + rd > D:
                                continue
                            if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                                continue
                            score = cz * 10000 + cy * 100 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Update coordinate sets
            cx_set.add(bx)
            cx_set.add(bx + bw)
            cy_set.add(by)
            cy_set.add(by + bh)
            cz_set.add(bz)
            cz_set.add(bz + bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
            
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_shake(boxes, container, initial_result, deadline):
    """
    Attempt to improve a placement by shaking (removing and re-placing) boxes.
    This is a local search improvement.
    """
    W, H, D = container
    n = len(boxes)
    
    # Copy result
    result = [list(r) for r in initial_result]
    
    # Build placed list from result
    placed_indices = [i for i in range(n) if result[i][0] == 0]
    
    if not placed_indices:
        return [tuple(r) for r in result]
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        valid_rots = [(rw, rh, rd) for rw, rh, rd in rots if rw <= W and rh <= H and rd <= D]
        if not valid_rots:
            valid_rots = [(w, h, d)]
        box_rotations.append(valid_rots)
    
    improved = True
    max_passes = 5
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        # Try to move each placed box to a better position
        # We iterate in reverse order of placement (last placed first) as they are more likely to be "loose"
        for i in reversed(placed_indices):
            if time.time() > deadline:
                break
            
            if result[i][0] != 0:
                continue
                
            # Remove box i
            x, y, z, w, h, d = result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]
            # We need to rebuild the placed list without this box for checking
            # For efficiency, we can just check against all other placed boxes
            
            # Generate candidates
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for j in range(n):
                if j != i and result[j][0] == 0:
                    px, py, pz, pw, ph, pd = result[j][1], result[j][2], result[j][3], result[j][4], result[j][5], result[j][6]
                    cx_set.add(px)
                    cx_set.add(px + pw)
                    cy_set.add(py)
                    cy_set.add(py + ph)
                    cz_set.add(pz)
                    cz_set.add(pz + pd)
            
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rots = box_rotations[i]
            
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        for rw, rh, rd in rots:
                            if cx + rw > W or cy + rh > H or cz + rd > D:
                                continue
                            
                            # Check overlap with all other placed boxes
                            overlap = False
                            for j in range(n):
                                if j == i or result[j][0] != 0:
                                    continue
                                px, py, pz, pw, ph, pd = result[j][1], result[j][2], result[j][3], result[j][4], result[j][5], result[j][6]
                                if not (cx + rw <= px or px + pw <= cx or
                                        cy + rh <= py or py + ph <= cy or
                                        cz + rd <= pz or pz + pd <= cz):
                                    overlap = True
                                    break
                            
                            if overlap:
                                continue
                            
                            # Score: prefer positions that are "lower" and "closer to origin"
                            # Also, prefer positions that don't change the bounding box of the packing?
                            # Simple score: z, y, x
                            score = cz * 10000 + cy * 100 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                # Check if this is an improvement
                # If the new position is "better" (lower score) than the old one, accept
                old_x, old_y, old_z = x, y, z
                old_score = old_z * 10000 + old_y * 100 + old_x
                new_score = best_score
                
                if new_score < old_score:
                    result[i] = [0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]]
                    improved = True
            # If no better position found, box stays where it is

    return [tuple(r) for r in result]

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Strategy 1: Volume-descending with ZYX candidate ordering
    # This is the primary strategy
    try:
        result1 = _try_place_optimized(boxes, container, sort_key='volume', candidate_order='zyx')
        filled1 = sum(r[4] * r[5] * r[6] for r in result1 if r[0] == 0)
    except Exception:
        result1 = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        filled1 = 0

    best_result = result1
    best_filled = filled1

    # If we have time, try other strategies
    if time.time() < deadline - 20.0:
        try:
            # Strategy 2: Max-dimension-descending with ZYX
            result2 = _try_place_optimized(boxes, container, sort_key='max_dim', candidate_order='zyx')
            filled2 = sum(r[4] * r[5] * r[6] for r in result2 if r[0] == 0)
            if filled2 > best_filled:
                best_result = result2
                best_filled = filled2
        except Exception:
            pass

    if time.time() < deadline - 20.0:
        try:
            # Strategy 3: Volume-descending with XYZ candidate ordering
            result3 = _try_place_optimized(boxes, container, sort_key='volume', candidate_order='xyz')
            filled3 = sum(r[4] * r[5] * r[6] for r in result3 if r[0] == 0)
            if filled3 > best_filled:
                best_result = result3
                best_filled = filled3
        except Exception:
            pass
            
    # If we have significant time left, try shaking the best result
    if time.time() < deadline - 10.0:
        try:
            result_shake = _try_place_shake(boxes, container, best_result, deadline - 2.0)
            filled_shake = sum(r[4] * r[5] * r[6] for r in result_shake if r[0] == 0)
            if filled_shake > best_filled:
                best_result = result_shake
                best_filled = filled_shake
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
