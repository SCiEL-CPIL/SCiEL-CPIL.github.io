import time
import numpy as np
from itertools import permutations

def _get_rotations(w, h, d):
    """Return unique rotations of a box."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _find_best_placement(box_dims, placed, W, H, D):
    """
    Find the best (BLB) placement for a box with given dimensions.
    Returns (x, y, z) or None if not found.
    """
    bw, bh, bd = box_dims
    
    # Generate candidate coordinates
    # Z candidates: 0 and top/bottom of placed boxes
    z_candidates = set([0])
    x_candidates = set([0])
    y_candidates = set([0])
    
    if len(placed) > 0:
        px, py, pz, pw, ph, pd = placed
        z_candidates.update(pz)
        z_candidates.update(pz + pd)
        x_candidates.update(px)
        x_candidates.update(px + pw)
        y_candidates.update(py)
        y_candidates.update(py + ph)
        
    # Filter candidates that fit within container
    z_candidates = sorted([z for z in z_candidates if z + bd <= D])
    x_candidates = sorted([x for x in x_candidates if x + bw <= W])
    y_candidates = sorted([y for y in y_candidates if y + bh <= H])
    
    if not z_candidates or not x_candidates or not y_candidates:
        return None

    # Vectorized overlap check setup
    if len(placed) > 0:
        px, py, pz, pw, ph, pd = placed
    else:
        px = np.array([])
        py = np.array([])
        pz = np.array([])
        pw = np.array([])
        ph = np.array([])
        pd = np.array([])

    # BLB: Minimize Z, then X, then Y
    for z in z_candidates:
        for x in x_candidates:
            for y in y_candidates:
                # Check overlap using vectorized ops
                if len(placed) > 0:
                    # Overlap if:
                    # x < px + pw AND x + bw > px
                    # y < py + ph AND y + bh > py
                    # z < pz + pd AND z + bd > pz
                    overlap_x = (x < px + pw) & (x + bw > px)
                    overlap_y = (y < py + ph) & (y + bh > py)
                    overlap_z = (z < pz + pd) & (z + bd > pz)
                    if not np.any(overlap_x & overlap_y & overlap_z):
                        return (x, y, z)
                else:
                    return (x, y, z)
                    
    return None

def _pack(boxes, container, sort_key):
    W, H, D = container
    # Sort boxes
    indexed_boxes = [(box, i) for i, box in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0]), reverse=True)
    
    placed = []
    result = [None] * len(boxes)
    
    for box, orig_idx in indexed_boxes:
        w, h, d = box
        rotations = _get_rotations(w, h, d)
        
        best_pos = None
        best_rot = None
        best_score = None
        
        for rot in rotations:
            pos = _find_best_placement(rot, placed, W, H, D)
            if pos is not None:
                # Score: BLB is already minimal. 
                # We can just take the first one found in the loop?
                # No, the loop in _find_best_placement returns the first valid one in Z, X, Y order.
                # So the first rotation that fits might not be the "best" if we consider other metrics?
                # BLB is deterministic. The first valid position found for a rotation is the BLB position for that rotation.
                # Should we prefer a specific rotation?
                # Usually, any valid BLB position is fine. 
                # However, different rotations lead to different BLB positions.
                # We should pick the rotation that yields the "best" position.
                # Since BLB minimizes Z, then X, then Y, we can compare the resulting (z, x, y) tuples.
                if best_pos is None:
                    best_pos = pos
                    best_rot = rot
                    best_score = (pos[2], pos[0], pos[1]) # z, x, y
                else:
                    current_score = (pos[2], pos[0], pos[1])
                    if current_score < best_score:
                        best_pos = pos
                        best_rot = rot
                        best_score = current_score
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            result[orig_idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    return result

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Strategies: Volume, Max Dim, Min Dim
    strategies = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: min(b)
    ]
    
    best_result = None
    best_volume = -1
    
    for i, strat in enumerate(strategies):
        if time.time() > deadline:
            break
            
        res = _pack(boxes, container, strat)
        
        # Calculate filled volume
        vol = 0
        for r in res:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
                
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
        # If we've filled a very high percentage, we can stop early?
        # Container volume
        W, H, D = container
        cont_vol = W * H * D
        if cont_vol > 0 and vol / cont_vol > 0.95:
            break

    if best_result is None:
        # Fallback: simple shelf packing (from seed)
        W, H, D = container
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
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
        best_result = placements

    return best_result
