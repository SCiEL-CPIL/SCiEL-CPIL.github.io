"""3D Bin Packing: multi-strategy greedy with First-Fit-Decreasing variants + local refinement."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _volume(w, h, d):
    return w * h * d

def _max_dim(w, h, d):
    return max(w, h, d)

def _sum_dim(w, h, d):
    return w + h + d

def _sorted_index(boxes, key_func):
    """Return indices sorted by key_func descending."""
    n = len(boxes)
    idx = list(range(n))
    idx.sort(key=lambda i: key_func(*boxes[i]), reverse=True)
    return idx

def _greedy_pack(boxes, container, order):
    """
    First-Fit-Decreasing style greedy: for each box in 'order', try to place it
    in the first bin where it fits using a simple corner-search heuristic.
    Returns list of (bin, x, y, z, w, h, d) or None if failed.
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bin_items:
            # Check overlap: intervals [x, x+w) and [ox, ox+ow) overlap if x < ox+ow and ox < x+w
            if x < ox + ow and ox < x + w and \
               y < oy + oh and oy < y + h and \
               z < oz + od and oz < z + d:
                return False
        return True
    
    def find_position(bin_items, w, h, d):
        """Find lowest-left-most position for box (w,h,d) in a bin."""
        # Generate candidate positions: origin + offsets from existing boxes
        candidates = [(0, 0, 0)]
        for (ox, oy, oz, ow, oh, od) in bin_items:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
        
        # Sort by (z, y, x) to prefer lowest, then leftmost, then backmost
        best = None
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if can_place(bin_items, cx, cy, cz, w, h, d):
                if best is None or (cz, cy, cx) < best:
                    best = (cz, cy, cx)
        return best
    
    for i in order:
        w, h, d = boxes[i]
        placed = False
        # Try existing bins first
        for b_idx in range(len(bins)):
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                cz, cy, cx = pos
                bins[b_idx].append((cx, cy, cz, w, h, d))
                placed = True
                break
        if not placed:
            # Open a new bin
            pos = find_position([], w, h, d)
            if pos is not None:
                cz, cy, cx = pos
                bins.append([(cx, cy, cz, w, h, d)])
                placed = True
            else:
                # Shouldn't happen if box fits in empty container
                return None
        if not placed:
            return None
    
    # Convert to result format
    placements = []
    for b_idx, bin_items in enumerate(bins):
        for (x, y, z, w, h, d) in bin_items:
            placements.append((b_idx, x, y, z, w, h, d))
    # Map back to original box indices
    # We need to track which original index went where
    # Let's redo with tracking
    placements = []
    for b_idx, bin_items in enumerate(bins):
        for item in bin_items:
            placements.append((b_idx, item[0], item[1], item[2], item[3], item[4], item[5]))
    return placements

def _greedy_pack_tracked(boxes, container, order):
    """
    First-Fit-Decreasing with tracking of original indices.
    Returns list of (bin, x, y, z, w, h, d) in original box order, or None.
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # list of lists of (x, y, z, w, h, d, orig_idx)
    
    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            if x < ox + ow and ox < x + w and \
               y < oy + oh and oy < y + h and \
               z < oz + od and oz < z + d:
                return False
        return True
    
    def find_position(bin_items, w, h, d):
        candidates = [(0, 0, 0)]
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
        
        best = None
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if can_place(bin_items, cx, cy, cz, w, h, d):
                if best is None or (cz, cy, cx) < best:
                    best = (cz, cy, cx)
        return best
    
    result_map = [None] * n  # (bin, x, y, z) for each original index
    
    for i in order:
        w, h, d = boxes[i]
        placed = False
        for b_idx in range(len(bins)):
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                cz, cy, cx = pos
                bins[b_idx].append((cx, cy, cz, w, h, d, i))
                result_map[i] = (b_idx, cx, cy, cz)
                placed = True
                break
        if not placed:
            pos = find_position([], w, h, d)
            if pos is not None:
                cz, cy, cx = pos
                bins.append([(cx, cy, cz, w, h, d, i)])
                result_map[i] = (len(bins) - 1, cx, cy, cz)
                placed = True
            else:
                return None
        if not placed:
            return None
    
    placements = []
    for i in range(n):
        b, x, y, z = result_map[i]
        w, h, d = boxes[i]
        placements.append((b, x, y, z, w, h, d))
    return placements

def _score_placements(placements, boxes, container):
    """Compute score: B + 0.1*(1 - last_bin_volume/container_volume)."""
    W, H, D = container
    cont_vol = W * H * D
    if cont_vol == 0:
        return 1e9
    if not placements:
        return 0.0
    
    bins = {}
    for (b, x, y, z, w, h, d) in placements:
        if b not in bins:
            bins[b] = 0
        bins[b] += w * h * d
    
    if not bins:
        return 0.0
    
    B = len(bins)
    max_bin = max(bins.keys())
    last_vol = bins.get(max_bin, 0)
    fraction = last_vol / cont_vol
    return B + 0.1 * (1 - fraction)

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Check all boxes fit in container
    for w, h, d in boxes:
        if w > W or h > H or d > D:
            # Box doesn't fit; return empty or best effort
            return []
    
    # Define sorting strategies
    strategies = [
        ("vol_desc", lambda i: _volume(*boxes[i])),
        ("vol_asc", lambda i: _volume(*boxes[i])),
        ("maxdim_desc", lambda i: _max_dim(*boxes[i])),
        ("maxdim_asc", lambda i: _max_dim(*boxes[i])),
        ("sumdim_desc", lambda i: _sum_dim(*boxes[i])),
        ("sumdim_asc", lambda i: _sum_dim(*boxes[i])),
        ("z_desc", lambda i: boxes[i][2]),
        ("y_desc", lambda i: boxes[i][1]),
        ("x_desc", lambda i: boxes[i][0]),
        ("ratio_desc", lambda i: _volume(*boxes[i]) / max(1, _max_dim(*boxes[i]))),
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
        
        order = _sorted_index(boxes, key_func)
        placements = _greedy_pack_tracked(boxes, container, order)
        if placements is None:
            continue
        
        score = _score_placements(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # If we haven't found any valid placement, use the seed shelf heuristic as fallback
    if best_placements is None:
        W2, H2, D2 = container
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W2:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H2:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D2:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    # Local refinement: try to reduce bins by moving boxes
    if time.time() < deadline - 5:
        W2, H2, D2 = container
        cont_vol = W2 * H2 * D2
        
        # Try to eliminate the last bin by moving its boxes to earlier bins
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            cur_placements = best_placements
            cur_score = _score_placements(cur_placements, boxes, container)
            
            # Find the last bin
            bins_used = set(p[0] for p in cur_placements)
            if not bins_used:
                break
            last_bin = max(bins_used)
            
            # Get boxes in last bin
            last_bin_indices = [i for i in range(n) if cur_placements[i][0] == last_bin]
            if not last_bin_indices:
                break
            
            # Try to move one box from last bin to an earlier bin
            moved = False
            for i in last_bin_indices:
                w, h, d = boxes[i]
                cur_b, cx, cy, cz = cur_placements[i][0], cur_placements[i][1], cur_placements[i][2], cur_placements[i][3]
                
                # Try placing in each earlier bin
                for b_target in sorted(bins_used - {last_bin}):
                    # Get all items in b_target
                    bin_items = []
                    for j in range(n):
                        if cur_placements[j][0] == b_target:
                            bin_items.append((cur_placements[j][1], cur_placements[j][2], cur_placements[j][3],
                                              boxes[j][0], boxes[j][1], boxes[j][2]))
                    
                    # Check if box i can be placed in b_target
                    # Remove its current position from consideration (it's in last_bin, not b_target)
                    def can_place_in(items, x, y, z, w, h, d):
                        if x + w > W2 or y + h > H2 or z + d > D2:
                            return False
                        for (ox, oy, oz, ow, oh, od) in items:
                            if x < ox + ow and ox < x + w and \
                               y < oy + oh and oy < y + h and \
                               z < oz + od and oz < z + d:
                                return False
                        return True
                    
                    # Find a position
                    candidates = [(0, 0, 0)]
                    for (ox, oy, oz, ow, oh, od) in bin_items:
                        candidates.append((ox + ow, oy, oz))
                        candidates.append((ox, oy + oh, oz))
                        candidates.append((ox, oy, oz + od))
                    
                    found_pos = None
                    for (px, py, pz) in candidates:
                        if can_place_in(bin_items, px, py, pz, w, h, d):
                            if found_pos is None or (pz, py, px) < found_pos:
                                found_pos = (pz, py, px)
                    
                    if found_pos is not None:
                        # Move box i to b_target
                        new_placements = list(cur_placements)
                        pz, py, px = found_pos
                        new_placements[i] = (b_target, px, py, pz, w, h, d)
                        
                        # Check if last bin is now empty
                        new_bins_used = set(p[0] for p in new_placements)
                        new_score = _score_placements(new_placements, boxes, container)
                        if new_score < cur_score:
                            best_placements = new_placements
                            best_score = new_score
                            moved = True
                            break
                
                if moved:
                    break
            
            if not moved:
                break
    
    return best_placements

# EVOLVE-BLOCK-END
