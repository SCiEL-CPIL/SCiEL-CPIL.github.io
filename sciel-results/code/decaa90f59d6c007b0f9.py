import time
import numpy as np
from itertools import permutations

def _all_rotations(dims):
    """Return all unique rotations of (w, h, d) as (w, h, d) tuples."""
    w, h, d = dims
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _fits(placed, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) doesn't overlap any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return False
    return True

def _place_box_greedy(boxes, container, order, deadline):
    """
    Place boxes in given order using greedy bottom-left-front with rotation search.
    Returns list of placements.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        bw, bh, bd = boxes[idx]
        rots = _all_rotations((bw, bh, bd))
        
        # Sort rotations by volume descending (all same volume), then by max dim ascending
        # Actually all rotations have same volume. Sort by (max_dim, min_dim) for better fit
        rots.sort(key=lambda r: (max(r), min(r)))
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        for rw, rh, rd in rots:
            # Try to find best position using a simple heuristic:
            # Scan candidate x positions, then y, then z
            # For efficiency, only check positions that are "supported" or at boundaries
            
            # Candidate x positions: 0 and right edges of placed boxes
            x_candidates = sorted(set([0] + [px + pw for (px, py, pz, pw, ph, pd) in placed if px + pw <= W]))
            x_candidates = [x for x in x_candidates if x + rw <= W]
            
            if not x_candidates:
                continue
            
            for cx in x_candidates:
                # Candidate y positions
                y_candidates = sorted(set([0] + [py + ph for (px, py, pz, pw, ph, pd) in placed if px < cx + rw and px + pw > cx and py + ph <= H]))
                y_candidates = [y for y in y_candidates if y + rh <= H]
                
                if not y_candidates:
                    continue
                
                for cy in y_candidates:
                    # Candidate z positions
                    z_candidates = sorted(set([0] + [pz + pd for (px, py, pz, pw, ph, pd) in placed if px < cx + rw and px + pw > cx and py < cy + rh and py + ph > cy and pz + pd <= D]))
                    z_candidates = [z for z in z_candidates if z + rd <= D]
                    
                    if not z_candidates:
                        continue
                    
                    for cz in z_candidates:
                        if _fits(placed, cx, cy, cz, rw, rh, rd):
                            # Score: prefer lower z, then lower y, then lower x
                            score = (cz, cy, cx)
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            w, h, d = best_rot
            placed.append((x, y, z, w, h, d))
            result[idx] = (0, x, y, z, w, h, d)
        else:
            result[idx] = (-1, 0, 0, 0, bw, bh, bd)
    
    # Fill any remaining None entries
    for i in range(len(boxes)):
        if result[i] is None:
            bw, bh, bd = boxes[i]
            result[i] = (-1, 0, 0, 0, bw, bh, bd)
    
    return result

def _score(placements, container):
    """Compute fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = 0
    for p in placements:
        if p[0] == 0:
            filled += p[4] * p[5] * p[6]
    return filled / total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Generate multiple ordering strategies
    strategies = []
    
    # Strategy 1: Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    strategies.append(vol_order)
    
    # Strategy 2: Max dimension descending
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)
    
    # Strategy 3: Volume/maxdim ratio (flatness)
    ratio_order = sorted(range(n), key=lambda i: (boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)
    strategies.append(ratio_order)
    
    # Strategy 4: Min dimension descending
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    strategies.append(mindim_order)
    
    # Strategy 5: Surface area descending
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(sa_order)
    
    # Strategy 6: Original order
    strategies.append(list(range(n)))
    
    best_placements = None
    best_score_val = -1
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        placements = _place_box_greedy(boxes, container, order, deadline)
        score_val = _score(placements, container)
        
        if score_val > best_score_val:
            best_score_val = score_val
            best_placements = placements
    
    # If no placements were made (shouldn't happen), return all left out
    if best_placements is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
