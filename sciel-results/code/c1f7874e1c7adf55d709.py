import numpy as np

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    # Sort indices by descending volume (largest first) for better packing
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bin_items:
            # Check overlap
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True
    
    def find_position(bin_items, w, h, d):
        # Try to find a valid position in this bin
        # Use a coarse-to-fine approach or just scan candidate positions
        # Candidate positions: 0, and edges of existing items
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (ox, oy, oz, ow, oh, od) in bin_items:
            xs.add(ox + ow)
            ys.add(oy + oh)
            zs.add(oz + od)
        
        # Also add some intermediate points if bins are sparse
        # Sort and filter to valid range
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        # If no candidates, try 0
        if not xs: xs = [0]
        if not ys: ys = [0]
        if not zs: zs = [0]
        
        # Try all combinations, prefer positions closer to origin (bottom-left-front)
        candidates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    if can_place(bin_items, x, y, z, w, h, d):
                        candidates.append((x, y, z))
        
        if not candidates:
            return None
        
        # Pick the candidate that minimizes (z, y, x) to pack toward the front-bottom-left
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        return candidates[0]
    
    for orig_idx, (w, h, d) in indexed:
        placed = False
        for bi in range(len(bins)):
            pos = find_position(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                placed = True
                break
        if not placed:
            # Open a new bin
            bins.append([(0, 0, 0, w, h, d)])
    
    # Build result
    placements = []
    for bi, bin_items in enumerate(bins):
        for (x, y, z, w, h, d) in bin_items:
            placements.append((bi, x, y, z, w, h, d))
    
    # We need to return in the original box order. Let's track original indices.
    # Rebuild: we need to map each placed box back to its original index.
    # Let's redo with tracking.
    
    return _pack_with_tracking(boxes, container)

def _pack_with_tracking(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d, orig_idx)
    
    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True
    
    def find_best_position(bin_items, w, h, d):
        # Find the position in this bin that minimizes x+y+z
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            xs.add(ox + ow)
            ys.add(oy + oh)
            zs.add(oz + od)
        
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        if not xs: xs = [0]
        if not ys: ys = [0]
        if not zs: zs = [0]
        
        best_pos = None
        best_score = float('inf')
        
        for x in xs:
            for y in ys:
                for z in zs:
                    if can_place(bin_items, x, y, z, w, h, d):
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            if score == 0:
                                return best_pos
        return best_pos
    
    result = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        best_bin_idx = -1
        best_pos = None
        global_best_score = float('inf')
        
        for bi in range(len(bins)):
            pos = find_best_position(bins[bi], w, h, d)
            if pos is not None:
                # Use position score to determine "best" fit
                # Lower score is better
                score = pos[0] + pos[1] + pos[2]
                if score < global_best_score:
                    global_best_score = score
                    best_bin_idx = bi
                    best_pos = pos
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d, orig_idx))
            result[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            result[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
    
    return result
