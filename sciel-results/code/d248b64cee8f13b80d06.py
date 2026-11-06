import numpy as np

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    # Sort indices by descending volume (largest first)
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d, orig_idx)
    
    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Check overlap
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True
    
    def find_best_position_in_bin(bin_items, w, h, d):
        # Generate candidate positions
        # Include 0 and the "faces" of existing boxes
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Add positions aligned with the "outer" faces
            xs.add(ox + ow)
            ys.add(oy + oh)
            zs.add(oz + od)
            # Add positions aligned with the "inner" faces (to pack tightly against them)
            xs.add(ox)
            ys.add(oy)
            zs.add(oz)
            
        # Filter valid candidates
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
                        # Heuristic: minimize distance to origin (compactness)
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            if score == 0:
                                return best_pos, 0
        return best_pos, best_score
    
    result = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        best_bin_idx = -1
        best_pos = None
        global_best_score = float('inf')
        
        # Best-Fit-Decreasing: Check all existing bins
        for bi in range(len(bins)):
            pos, score = find_best_position_in_bin(bins[bi], w, h, d)
            if pos is not None:
                if score < global_best_score:
                    global_best_score = score
                    best_bin_idx = bi
                    best_pos = pos
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d, orig_idx))
            result[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            result[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
    return result
