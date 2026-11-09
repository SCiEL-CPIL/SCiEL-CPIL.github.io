import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Precompute box volumes for scoring
    box_volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Sort boxes by volume descending for better packing (but we need to return in original order)
    # Actually, we need to place all boxes. Let's use a best-fit strategy.
    # We'll maintain a list of bins, each with a 3D numpy array for occupancy.
    # For efficiency, we'll try to find the best position in existing bins first, then open new bins.
    
    # To keep it fast, we limit the number of bins we consider and use a heuristic search.
    # Since 40 boxes and 120s limit, we can be somewhat exhaustive per box.
    
    # Let's use a simpler but effective approach:
    # 1. Sort boxes by some heuristic (e.g., max dimension or volume) to pack large ones first.
    # 2. For each box, try to find the best position in existing bins.
    # 3. "Best" means minimizing the sum of distances to the nearest occupied cell in each dimension,
    #    or simply the lowest z, then y, then x (bottom-left-front corner).
    
    # Actually, the directive suggests using a 3D grid-based occupancy map and checking
    # np.any(occupied[x:x+w, y:y+h, z:z+d]) for feasibility.
    # Let's implement a "first-fit decreasing" with a best-fit position search.
    
    # We'll sort boxes by volume descending, but remember original indices.
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # Each bin is represented by:
    # - occ: 3D boolean numpy array of shape (W, H, D)
    # - placements: list of (x, y, z, w, h, d)
    
    bins = []  # list of dicts: {'occ': np.ndarray, 'placements': list}
    
    def new_bin():
        return {'occ': np.zeros((W, H, D), dtype=bool), 'placements': []}
    
    def find_best_position(occ, w, h, d):
        """
        Find the best (x, y, z) position for a box of size (w, h, d) in a bin with occupancy 'occ'.
        Returns (x, y, z) or None if no position is feasible.
        Heuristic: minimize (z, y, x) - i.e., prefer bottom, then front, then left.
        For efficiency, we iterate z from 0 to D-d, y from 0 to H-h, x from 0 to W-w.
        Since W, H, D could be large, this might be slow. But for 40 boxes, it should be okay.
        """
        best = None
        best_score = None
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    # Check if this position is free
                    if not np.any(occ[x:x+w, y:y+h, z:z+d]):
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best = (x, y, z)
                            best_score = score
                            # Since we're iterating z, y, x in increasing order,
                            # the first feasible position is the best by our heuristic.
                            return best
        return None
    
    def place_box_in_bin(bin_dict, w, h, d):
        pos = find_best_position(bin_dict['occ'], w, h, d)
        if pos is None:
            return False
        x, y, z = pos
        bin_dict['occ'][x:x+w, y:y+h, z:z+d] = True
        bin_dict['placements'].append((x, y, z, w, h, d))
        return True
    
    # Process boxes in sorted order
    box_placements = [None] * n  # (bin_index, x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        for b_idx in range(len(bins)):
            if place_box_in_bin(bins[b_idx], w, h, d):
                x, y, z, _, _, _ = bins[b_idx]['placements'][-1]
                box_placements[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            # Open a new bin
            bins.append(new_bin())
            b_idx = len(bins) - 1
            place_box_in_bin(bins[b_idx], w, h, d)
            x, y, z, _, _, _ = bins[b_idx]['placements'][-1]
            box_placements[idx] = (b_idx, x, y, z, w, h, d)
    
    # Convert to the required format: (bin, x, y, z, w, h, d)
    # Bin labels are nonnegative integers. The largest label defines the last bin.
    # We use b_idx as the bin label.
    
    result = []
    for i in range(n):
        if box_placements[i] is not None:
            result.append(box_placements[i])
    
    return result
