"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import math
import heapq
import itertools

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending (FFD)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins: list of lists of (x, y, z, w, h, d)
    bins = []
    
    def check_overlap(x, y, z, w, h, d, existing):
        for ex, ey, ez, ew, eh, ed in existing:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False
        return False

    def find_place(bin_items, w, h, d):
        # Generate candidate positions from corners of existing items
        xs = {0}
        ys = {0}
        zs = {0}
        for ex, ey, ez, ew, eh, ed in bin_items:
            if ex + ew <= W:
                xs.add(ex + ew)
            if ey + eh <= H:
                ys.add(ey + eh)
            if ez + ed <= D:
                zs.add(ez + ed)
        
        # Also add positions that align with the right/top/back of each item
        # This helps fill gaps
        for ex, ey, ez, ew, eh, ed in bin_items:
            # Positions to the right, above, and behind each item
            if ex + ew + w <= W:
                xs.add(ex + ew)
            if ey + eh + h <= H:
                ys.add(ey + eh)
            if ez + ed + d <= D:
                zs.add(ez + ed)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_pos = None
        best_score = float('inf')
        
        # Prefer positions that minimize (x, y, z) lexicographically,
        # but also consider how much "waste" is created
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if not check_overlap(x, y, z, w, h, d, bin_items):
                        # Score: prefer lower x, then y, then z
                        # Weight x most heavily since it's the primary packing direction
                        score = x * (H * D) + y * D + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos

    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place remaining boxes in new bins at origin
            # This is a last resort to ensure we return something
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    # Try to place in an existing bin quickly
                    placed = False
                    for b_idx in range(len(bins)):
                        if check_overlap(0, 0, 0, rw, rh, rd, bins[b_idx]):
                            continue
                        # Quick check if it fits at origin
                        if rw <= W and rh <= H and rd <= D:
                            bins[b_idx].append((0, 0, 0, rw, rh, rd))
                            placements[rem_idx] = (b_idx, 0, 0, 0, rw, rh, rd)
                            placed = True
                            break
                    if not placed:
                        # New bin
                        bins.append([(0, 0, 0, rw, rh, rd)])
                        placements[rem_idx] = (len(bins) - 1, 0, 0, 0, rw, rh, rd)
            return placements

        placed = False
        
        # Try existing bins in order (first-fit decreasing)
        for b_idx in range(len(bins)):
            pos = find_place(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Start a new bin
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in any bin - should not happen for valid inputs
                placements[orig_idx] = (len(bins), 0, 0, 0, w, h, d)
                bins.append([])
    
    return placements
# EVOLVE-BLOCK-END
