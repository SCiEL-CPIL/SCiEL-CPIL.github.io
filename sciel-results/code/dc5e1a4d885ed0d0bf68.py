import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Deadline guard: 115s out of 120s limit
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Preprocess: sort boxes by volume descending, then by max dimension
    # This helps pack large boxes first
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1])), reverse=True)
    
    # For each box, generate all unique orientations
    def get_orientations(w, h, d):
        dims = sorted([w, h, d])
        orients = set()
        for perm in itertools.permutations(dims):
            orients.add(perm)
        return list(orients)
    
    # 3D occupancy grid approach with binary search on z
    # We'll use a grid-based approach where we track occupied cells
    # But for efficiency, we'll use a smarter approach: track "free space" as rectangles/boxes
    
    # Actually, let's use a simpler but effective approach:
    # 1. Sort boxes by size (largest first)
    # 2. For each box, try all orientations
    # 3. For each orientation, find the lowest (z, y, x) position where it fits
    # 4. Use a 3D boolean grid for occupancy check
    
    # Grid resolution: use unit grid for simplicity, but this may be too slow
    # Instead, let's use a list of placed boxes and check overlap
    
    placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    placed = []  # list of (x, y, z, w, h, d)
    
    # Try to place each box
    for idx, (orig_idx, (bw, bh, bd)) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
            
        orients = get_orientations(bw, bh, bd)
        best_pos = None
        best_volume = -1
        
        # For each orientation, try to find the best position
        for (ow, oh, od) in orients:
            if ow > W or oh > H or od > D:
                continue
                
            # Try to place at various positions
            # Strategy: try bottom-left-front first, then scan
            # Use a simple heuristic: try positions in order of (z, y, x)
            
            # First, try the "natural" position: (0, 0, 0)
            # Then scan upward in z, then y, then x
            
            # For efficiency, let's try a limited set of candidate positions
            # Based on existing placements
            
            # Generate candidate x, y, z positions
            # x candidates: 0, and right edges of placed boxes
            # y candidates: 0, and front edges of placed boxes  
            # z candidates: 0, and top edges of placed boxes
            
            x_cands = [0]
            y_cands = [0]
            z_cands = [0]
            
            for (px, py, pz, pw, ph, pd) in placed:
                x_cands.append(px + pw)
                y_cands.append(py + ph)
                z_cands.append(pz + pd)
            
            x_cands = sorted(set(x for x in x_cands if x + ow <= W))
            y_cands = sorted(set(y for y in y_cands if y + oh <= H))
            z_cands = sorted(set(z for z in z_cands if z + od <= D))
            
            # Try all combinations (limited by the number of candidates)
            # To keep it fast, limit the search
            if len(x_cands) > 20:
                x_cands = x_cands[:20]
            if len(y_cands) > 20:
                y_cands = y_cands[:20]
            if len(z_cands) > 20:
                z_cands = z_cands[:20]
            
            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        # Check if this position is free
                        if _is_free(x, y, z, ow, oh, od, placed):
                            # Score: prefer lower z, then lower y, then lower x
                            score = z * W * H + y * W + x
                            if best_pos is None or score < best_volume:
                                best_pos = (x, y, z, ow, oh, od)
                                best_volume = score
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            # Also try a simple scan from origin if no position found
            if best_pos is None:
                # Try placing at (0, 0, 0) and scan
                for z in range(0, D - od + 1, max(1, od // 4)):
                    for y in range(0, H - oh + 1, max(1, oh // 4)):
                        for x in range(0, W - ow + 1, max(1, ow // 4)):
                            if _is_free(x, y, z, ow, oh, od, placed):
                                best_pos = (x, y, z, ow, oh, od)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
        
        if best_pos is not None:
            x, y, z, ow, oh, od = best_pos
            placed.append((x, y, z, ow, oh, od))
            placements[orig_idx] = (0, x, y, z, ow, oh, od)
        else:
            # Box cannot be placed
            placements[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
    
    return placements


def _is_free(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dimensions (w,h,d) doesn't overlap any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap
        if x < px + pw and x + w > px and \
           y < py + ph and y + h > py and \
           z < pz + pd and z + d > pz:
            return False
    return True
