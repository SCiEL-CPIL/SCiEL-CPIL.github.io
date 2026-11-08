"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import bisect

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins structure: list of lists of placements (x, y, z, w, h, d)
    bins = []
    
    def check_overlap(x, y, z, w, h, d, existing):
        for ex, ey, ez, ew, eh, ed in existing:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False

    def find_place(bin_items, w, h, d):
        # Try to find a spot in the current bin
        # Simple approach: iterate through grid or sorted candidates
        # For performance, we can maintain a sorted list of "free" x-coordinates
        # But for simplicity and correctness first, we use a heuristic scan.
        
        # Get unique x, y, z coordinates from existing items to define candidate positions
        xs = {0}
        ys = {0}
        zs = {0}
        for ex, ey, ez, ew, eh, ed in bin_items:
            if ex + ew <= W: xs.add(ex + ew)
            if ey + eh <= H: ys.add(ey + eh)
            if ez + ed <= D: zs.add(ez + ed)
        
        # Sort coordinates to prioritize lower values (packing towards origin)
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_pos = None
        best_score = float('inf')
        
        for x in xs:
            if x + w > W: continue
            for y in ys:
                if y + h > H: continue
                for z in zs:
                    if z + d > D: continue
                    if not check_overlap(x, y, z, w, h, d, bin_items):
                        # Score: prefer lower x, then y, then z
                        score = x * 10000 + y * 100 + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            
        return best_pos

    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place remaining boxes in a new bin each or force place
            # This should rarely happen, but ensure we return something valid
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    placements[rem_idx] = (9999, 0, 0, 0, rw, rh, rd)
            return placements

        placed = False
        # Try existing bins
        for b_idx in range(len(bins)):
            pos = find_place(bins[b_idx], w, h, d)
            if pos:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Start a new bin
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in any bin (should not happen if container is valid for all)
                placements[orig_idx] = (len(bins), 0, 0, 0, w, h, d)
                bins.append([]) # Dummy to keep index consistent? No, just handle gracefully.
                # Actually, if it doesn't fit, we can't place it validly. 
                # But task implies all boxes fit.
                
    return placements
# EVOLVE-BLOCK-END
