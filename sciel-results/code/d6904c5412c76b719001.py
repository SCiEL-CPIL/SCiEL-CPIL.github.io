import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Improved 3D bin packing using Best-Fit Decreasing with enhanced candidate generation.
    
    Strategy:
    1. Sort boxes by volume descending, with tie-breaking by max dimension.
    2. For each box, evaluate all existing bins and find the best position
       (minimizing wasted space) using comprehensive candidate points.
    3. If no existing bin fits, start a new bin.
    4. Uses deadline guard for safety.
    
    Returns:
    List of placements: (bin_id, x, y, z, w, h, d)
    """
    W, H, D = container
    container_vol = W * H * D
    
    # Deadline guard
    start_time = time.time()
    timeout = 120  # 120 seconds hard limit
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1])), reverse=True)
    
    bins = [] # List of lists of (x,y,z,w,h,d)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def get_bin_fill_ratio(bin_boxes):
        total_vol = 0
        for _, _, _, bw, bh, bd in bin_boxes:
            total_vol += bw * bh * bd
        return total_vol / container_vol

    def compute_waste(bin_boxes, x, y, z, w, h, d):
        # Compute the increase in bounding box volume if we place this box here
        if not bin_boxes:
            return w * h * d  # Bounding box is just the box itself
        
        max_x = max(bx + bw for bx, by, bz, bw, bh, bd in bin_boxes)
        max_y = max(by + bh for bx, by, bz, bw, bh, bd in bin_boxes)
        max_z = max(bz + bd for bx, by, bz, bw, bh, bd in bin_boxes)
        
        new_max_x = max(max_x, x + w)
        new_max_y = max(max_y, y + h)
        new_max_z = max(max_z, z + d)
        
        current_bounding_vol = max_x * max_y * max_z
        new_bounding_vol = new_max_x * new_max_y * new_max_z
        
        return new_bounding_vol - current_bounding_vol

    def find_best_position(bin_boxes, w, h, d):
        """Find the position that minimizes waste for a given bin."""
        if not bin_boxes:
            if can_place(bin_boxes, 0, 0, 0, w, h, d):
                return (0, 0, 0)
            return None

        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidate points from existing boxes
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Direct neighbors
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            
            # Diagonal combinations
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Floor-aligned candidates
            candidates.add((bx + bw, by, 0))
            candidates.add((bx, by + bh, 0))
            candidates.add((0, by, bz + bd))
            candidates.add((bx, 0, bz + bd))
            candidates.add((bx + bw, 0, 0))
            candidates.add((0, by + bh, 0))
            candidates.add((0, 0, bz + bd))
            
            # Cross-axis combinations with other boxes
            for bx2, by2, bz2, bw2, bh2, bd2 in bin_boxes:
                if bx2 == bx and by2 == by and bz2 == bz:
                    continue
                # Try x from one box, y from another
                candidates.add((bx + bw, by2, bz))
                candidates.add((bx, by2 + bh2, bz))
                candidates.add((bx + bw, by2, bz2 + bd2))
        
        # Filter and sort candidates
        valid_candidates = []
        for x, y, z in candidates:
            if x >= W or y >= H or z >= D:
                continue
            if x + w > W or y + h > H or z + d > D:
                continue
            if can_place(bin_boxes, x, y, z, w, h, d):
                waste = compute_waste(bin_boxes, x, y, z, w, h, d)
                # Prioritize positions with lower waste, then lower coordinates
                valid_candidates.append((waste, x, y, z, (x, y, z)))
        
        if not valid_candidates:
            return None
        
        # Sort by waste first, then by position compactness
        valid_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return valid_candidates[0][4]

    # Store results in original order
    results = [None] * len(boxes)
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() - start_time > timeout - 5:
            # Fast fallback: place in new bins quickly
            for remaining_idx, (rw, rh, rd) in indexed_boxes:
                if results[remaining_idx] is None:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    results[remaining_idx] = (b_idx, 0, 0, 0, rw, rh, rd)
            break

        placed = False
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        # Try existing bins and find the one with minimum waste
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                waste = compute_waste(bins[b_idx], x, y, z, w, h, d)
                # Use fill ratio as secondary criterion: prefer bins with higher fill
                fill_ratio = get_bin_fill_ratio(bins[b_idx])
                # Combined score: minimize waste, then prefer higher fill ratio
                combined_score = waste - 200 * fill_ratio
                if combined_score < best_waste:
                    best_waste = combined_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            results[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Start new bin
            b_idx = len(bins)
            x, y, z = 0, 0, 0
            bins.append([(x, y, z, w, h, d)])
            results[idx] = (b_idx, x, y, z, w, h, d)

    return results

# EVOLVE-BLOCK-END
