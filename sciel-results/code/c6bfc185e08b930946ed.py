"""3D Container Loading: Best-Fit greedy with multiple heuristics + gap-filling local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import defaultdict

def _orientations(w, h, d):
    """Return all 6 distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes strictly overlap."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _score_position(x, y, z, w, h, d, W, H, D, cur_max_x, cur_max_y, cur_max_z):
    """
    Score a position for best-fit: minimize the volume of the bounding box.
    """
    new_max_x = max(cur_max_x, x + w)
    new_max_y = max(cur_max_y, y + h)
    new_max_z = max(cur_max_z, z + d)
    return new_max_x * new_max_y * new_max_z

def _pack_best_fit(boxes, container, order, deadline):
    """
    Pack boxes using Best-Fit heuristic.
    For each box, try all orientations.
    For each orientation, try all candidate positions.
    Select the position that minimizes the resulting bounding box volume.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    # Current bounding box
    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Pre-compute candidate coordinates
        xs = {0}
        ys = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
        
        xs = sorted(x for x in xs if 0 <= x <= W)
        ys = sorted(y for y in ys if 0 <= y <= H)
        
        # For each orientation
        for ow, oh, od in _orientations(w, h, d):
            # For each candidate X, Y
            for cx in xs:
                if cx + ow > W:
                    continue
                for cy in ys:
                    if cy + oh > H:
                        continue
                    
                    # Find valid Z candidates for this (cx, cy)
                    z_cands = {0}
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Check if box (px, py, pz, pw, ph, pd) overlaps with (cx, cy, ?, ow, oh, ?) in XY
                        if cx + ow > px and px + pw > cx and cy + oh > py and py + ph > cy:
                            z_cands.add(pz + pd)
                    
                    z_cands = sorted(z for z in z_cands if 0 <= z <= D)
                    
                    for cz in z_cands:
                        if cz + od > D:
                            continue
                        
                        # Check if (cx, cy, cz, ow, oh, od) fits (no overlap)
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            # Calculate BB volume
                            score = _score_position(cx, cy, cz, ow, oh, od, W, H, D, cur_max_x, cur_max_y, cur_max_z)
                            
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
            # Update current max
            cur_max_x = max(cur_max_x, cx + ow)
            cur_max_y = max(cur_max_y, cy + oh)
            cur_max_z = max(cur_max_z, cz + od)
            
    return placements, filled_vol

def _local_search(boxes, container, placements, filled_vol, deadline):
    """
    Local search: try to re-insert each box at a better position using Best-Fit scoring.
    This can help fix suboptimal greedy decisions.
    """
    W, H, D = container
    n = len(boxes)
    best_placements = dict(placements)
    best_filled = filled_vol
    
    # Build initial placed list
    placed = []
    for idx in sorted(best_placements.keys()):
        x, y, z, w, h, d = best_placements[idx]
        placed.append((x, y, z, w, h, d))
    
    # We will do a few passes of re-insertion
    for _iteration in range(2):  # Limit passes to avoid timeout
        if time.time() > deadline:
            break
        
        improved = False
        # Process boxes in reverse order of insertion (or random) to break dependencies
        # Using reverse of the original order might be better? Or just iterate keys.
        # Let's iterate over a copy of the keys.
        for idx in list(best_placements.keys()):
            if time.time() > deadline:
                break
            
            # Remove box from placements
            bx, by, bz, bw, bh, bd = best_placements[idx]
            # Remove from placed list
            placed = [p for p in placed if p != (bx, by, bz, bw, bh, bd)]
            del best_placements[idx]
            
            # Recalculate current max extents (expensive but necessary for correct BB scoring)
            cur_max_x = 0
            cur_max_y = 0
            cur_max_z = 0
            for (px, py, pz, pw, ph, pd) in placed:
                cur_max_x = max(cur_max_x, px + pw)
                cur_max_y = max(cur_max_y, py + ph)
                cur_max_z = max(cur_max_z, pz + pd)
            
            w, h, d = boxes[idx]
            best_score = float('inf')
            best_pos = None
            best_orient = None
            
            # Pre-compute candidate coordinates
            xs = {0}
            ys = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                xs.add(px)
                xs.add(px + pw)
                ys.add(py)
                ys.add(py + ph)
            
            xs = sorted(x for x in xs if 0 <= x <= W)
            ys = sorted(y for y in ys if 0 <= y <= H)
            
            # For each orientation
            for ow, oh, od in _orientations(w, h, d):
                # For each candidate X, Y
                for cx in xs:
                    if cx + ow > W:
                        continue
                    for cy in ys:
                        if cy + oh > H:
                            continue
                        
                        # Find valid Z candidates for this (cx, cy)
                        z_cands = {0}
                        for (px, py, pz, pw, ph, pd) in placed:
                            # Check if box (px, py, pz, pw, ph, pd) overlaps with (cx, cy, ?, ow, oh, ?) in XY
                            if cx + ow > px and px + pw > cx and cy + oh > py and py + ph > cy:
                                z_cands.add(pz + pd)
                        
                        z_cands = sorted(z for z in z_cands if 0 <= z <= D)
                        
                        for cz in z_cands:
                            if cz + od > D:
                                continue
                            
                            # Check if (cx, cy, cz, ow, oh, od) fits (no overlap)
                            if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                                # Calculate BB volume
                                score = _score_position(cx, cy, cz, ow, oh, od, W, H, D, cur_max_x, cur_max_y, cur_max_z)
                                
                                if score < best_score:
                                    best_score = score
                                    best_pos = (cx, cy, cz)
                                    best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                # Check if this is better than current (or if it's a new placement)
                # Since we removed it, if it fits, it's an improvement or same.
                # We accept it if it changes position or orientation, or if it's just re-inserted.
                # Actually, we just want to see if we can pack more boxes.
                # If we remove a box and can't re-insert it, we've made things worse.
                # So we should only re-insert if we find a valid spot.
                # If we find a valid spot, we place it.
                # Does it improve the total volume? No, volume is the same.
                # Does it improve the bounding box? Maybe.
                # But the main goal is to fit MORE boxes.
                # If we remove a box, we free up space. If we re-insert it, we use that space.
                # This doesn't help fit new boxes unless the new position allows other boxes to fit.
                # This is a "shake" move.
                
                best_placements[idx] = (cx, cy, cz, ow, oh, od)
                placed.append((cx, cy, cz, ow, oh, od))
                improved = True
            else:
                # Could not re-place; revert
                best_placements[idx] = (bx, by, bz, bw, bh, bd)
                placed.append((bx, by, bz, bw, bh, bd))
        
        if not improved:
            break
    
    return best_placements, best_filled

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # Generate multiple sort orders
    orders = []
    
    # Order 1: by volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # Order 2: by max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # Order 3: by sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # Order 4: by aspect ratio (most elongated first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(min(boxes[i]), 1e-9)), reverse=True))
    
    # Order 5: by min dimension descending (flat boxes first)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    best_result = None
    best_filled = 0.0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        placements, filled_vol = _pack_best_fit(boxes, container, order, deadline)
        
        # Apply local search to improve
        if time.time() < deadline:
            placements, filled_vol = _local_search(boxes, container, placements, filled_vol, deadline)
        
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements
    
    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
