import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with enhanced candidate generation
    and a post-processing consolidation pass to reduce bin count.
    
    Strategy:
    1. Sort boxes by volume descending, with tie-breaking by max dimension.
    2. For each box, evaluate all existing bins and find the best position
       (minimizing wasted space) using comprehensive candidate points.
    3. If no existing bin fits, start a new bin.
    4. Post-processing: try to consolidate by moving boxes from higher-numbered
       bins into lower-numbered bins to reduce total bin count.
    5. Uses deadline guard for safety.
    
    Returns:
    List of placements: (bin_id, x, y, z, w, h, d) in original input order
    """
    W, H, D = container
    container_vol = W * H * D
    
    # Deadline guard with safety margin
    start_time = time.time()
    timeout = 130  # Safety margin before 150s kill
    
    # Sort boxes: primary by volume desc, secondary by max dimension desc
    indexed_boxes = sorted(
        enumerate(boxes), 
        key=lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1])), 
        reverse=True
    )
    
    bins = []  # List of lists of (x,y,z,w,h,d)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box can be placed at (x,y,z) in given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def compute_waste(bin_boxes, x, y, z, w, h, d):
        """Compute a waste metric to prefer tighter fits.
        Uses strict inequality to avoid counting zero-gap as waste."""
        if not bin_boxes:
            return 0.0
        waste = 0.0
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Gap in x direction
            if x + w < bx:
                waste += (bx - x - w) * h * d
            elif bx + bw < x:
                waste += (x - bx - bw) * h * d
            # Gap in y direction
            if y + h < by:
                waste += w * (by - y - h) * d
            elif by + bh < y:
                waste += w * (y - by - bh) * d
            # Gap in z direction
            if z + d < bz:
                waste += w * h * (bz - z - d)
            elif bz + bd < z:
                waste += w * h * (z - bz - bd)
        return waste

    def find_best_position(bin_boxes, w, h, d):
        """Find the best position in this bin, or None if can't fit."""
        if not bin_boxes:
            if can_place(bin_boxes, 0, 0, 0, w, h, d):
                return (0, 0, 0)
            return None

        candidates = set()
        
        # Always try origin
        candidates.add((0, 0, 0))
        
        # Generate candidate points from box faces and corners
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Face-adjacent positions
            candidates.add((bx + bw, by, bz))      # Right of box
            candidates.add((bx, by + bh, bz))      # Above box in y
            candidates.add((bx, by, bz + bd))      # Above box in z
            
            # Corner combinations
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            
            # Floor-aligned positions
            candidates.add((bx + bw, by, 0))
            candidates.add((bx, by + bh, 0))
            candidates.add((bx + bw, 0, bz))
            candidates.add((0, by, bz + bd))
            candidates.add((bx, 0, bz + bd))
            candidates.add((bx + bw, 0, 0))
            candidates.add((0, by + bh, 0))
            candidates.add((0, 0, bz + bd))
            
            # Additional structural candidates
            candidates.add((bx, by + bh, 0))
            candidates.add((bx + bw, by, 0))
            candidates.add((0, by, bz))
            candidates.add((bx, by, 0))

        # Filter valid candidates and sort by priority
        valid_candidates = []
        for x, y, z in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                if can_place(bin_boxes, x, y, z, w, h, d):
                    # Priority: prefer positions closer to origin, then lower waste
                    dist = x + y + z
                    waste = compute_waste(bin_boxes, x, y, z, w, h, d)
                    valid_candidates.append((dist, waste, x, y, z))
        
        if not valid_candidates:
            return None
        
        # Sort by distance first, then waste
        valid_candidates.sort()
        return (valid_candidates[0][2], valid_candidates[0][3], valid_candidates[0][4])

    def get_bin_fill_ratio(bin_boxes):
        """Calculate fill ratio of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(bw * bh * bd for _, _, _, bw, bh, bd in bin_boxes)
        return total_vol / container_vol

    # Store results in original order
    results = [None] * len(boxes)
    
    for idx, (w, h, d) in indexed_boxes:
        # Check timeout - if close to limit, use fast fallback
        elapsed = time.time() - start_time
        if elapsed > timeout - 2:
            # Emergency: place in new bin immediately
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0, w, h, d)
            continue

        placed = False
        
        # Best-fit: find the bin with the best position (lowest waste)
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: prefer bins with higher fill ratio, and lower waste
                fill = get_bin_fill_ratio(bins[b_idx])
                waste = compute_waste(bins[b_idx], x, y, z, w, h, d)
                # Combined score: minimize waste, prefer filled bins
                score = waste - 200.0 * fill  # Strong preference for filled bins
                if score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx >= 0:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            results[idx] = (best_bin_idx, x, y, z, w, h, d)
            placed = True
        
        if not placed:
            # Start new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0, w, h, d)

    # Post-processing: try to consolidate bins by moving boxes from higher to lower bins
    # This can reduce the total number of bins used
    elapsed = time.time() - start_time
    if elapsed < timeout - 10 and len(bins) > 1:
        # Try to move boxes from the last bin into earlier bins
        # Process bins from highest to lowest, trying to empty each one
        for src_bin in range(len(bins) - 1, -1, -1):
            if elapsed > timeout - 5:
                break
            if not bins[src_bin]:
                continue
            
            # Try to move each box in this bin to an earlier bin
            moved_any = True
            while moved_any and bins[src_bin]:
                moved_any = False
                for box_idx in range(len(bins[src_bin])):
                    if elapsed > timeout - 5:
                        break
                    x, y, z, w, h, d = bins[src_bin][box_idx]
                    
                    # Try to place this box in each earlier bin
                    for dst_bin in range(src_bin):
                        if not bins[dst_bin]:
                            continue
                        pos = find_best_position(bins[dst_bin], w, h, d)
                        if pos is not None:
                            # Move the box
                            bins[dst_bin].append((pos[0], pos[1], pos[2], w, h, d))
                            bins[src_bin].pop(box_idx)
                            moved_any = True
                            break
                    if not moved_any:
                        break
            
            # If bin is now empty, remove it and renumber
            if not bins[src_bin]:
                bins.pop(src_bin)
                # Renumber all results
                for i in range(len(results)):
                    if results[i] is not None:
                        old_bin = results[i][0]
                        if old_bin > src_bin:
                            results[i] = (old_bin - 1, results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6])
                        elif old_bin == src_bin:
                            # This shouldn't happen since bin is empty
                            pass

    return results

# EVOLVE-BLOCK-END
