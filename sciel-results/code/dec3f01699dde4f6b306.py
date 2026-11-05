"""3D Bin Packing: Best-fit decreasing with efficient candidate generation and bin selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a bin.
        Returns (x, y, z) or None if no valid position exists.
        Uses candidate points from box faces and scores by "tightness".
        """
        if not bin_boxes:
            # Empty bin: place at origin
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate coordinates from existing box faces
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer positions that minimize the "envelope"
                        # Use a weighted sum that encourages compact packing
                        # Higher weight on z (depth) to pack toward front
                        score = 0.3*x + 0.3*y + 0.4*z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization and tighter positions
                util = bin_utilization(bin_boxes)
                pos_score = 0.3*x + 0.3*y + 0.4*z
                # We want to minimize pos_score and maximize util
                # So: minimize pos_score - lambda * util
                score = pos_score - 0.5 * util
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
