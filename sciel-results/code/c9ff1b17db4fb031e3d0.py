import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    # We need to keep track of original indices to return correct placements
    # Actually, the return format is (bin, x, y, z, w, h, d) per box in original order.
    # So we process a sorted list of (index, w, h, d) and then map back.
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # State: list of bins. Each bin is a list of free spaces (maximal empty cuboids).
    # A free space is (x, y, z, w, h, d) representing a box at (x,y,z) with size (w,h,d).
    # Initially, one bin with one free space: the entire container.
    bins = []
    def new_bin():
        # Free space covers entire container
        return [(0, 0, 0, W, H, D)]
    
    bins.append(new_bin())
    
    # Placement results: list of (bin_idx, x, y, z, w, h, d) for each original box index
    placements = [None] * n
    
    # Helper: Check if a box (w,h,d) fits in a free space (fx, fy, fz, fw, fh, fd)
    def fits_in_space(fx, fy, fz, fw, fh, fd, w, h, d):
        return w <= fw and h <= fh and d <= fd

    # Helper: Split a free space after placing a box at (px, py, pz) with size (w, h, d)
    # Returns list of new free spaces
    def split_space(fx, fy, fz, fw, fh, fd, px, py, pz, w, h, d):
        new_spaces = []
        
        # Check if box fits exactly
        if w == fw and h == fh and d == fd:
            return []
            
        # Split along X if there is space
        if px + w < fx + fw:
            # Left part (actually right part relative to box start, but box starts at px)
            # The space is from fx to fx+fw. Box is from px to px+w.
            # Since px >= fx, the part to the left of the box (if any) is fx to px.
            # The part to the right of the box is px+w to fx+fw.
            # We only generate the part to the right if it's non-empty.
            # Wait, the standard maximal space partitioning splits into 6 parts.
            # 1. Left: (fx, fy, fz, px-fx, fh, fd) if px > fx
            # 2. Right: (px+w, fy, fz, fx+fw-(px+w), fh, fd) if px+w < fx+fw
            # 3. Below: (px, fy, fz, w, py-py, d) -> (px, fy, fz, w, py-fy, d) if py > fy
            # 4. Above: (px, py+h, fz, w, fy+fh-(py+h), d) if py+h < fy+fh
            # 5. Front: (px, py, fz, w, h, pz-fz) if pz > fz
            # 6. Back: (px, py, pz+d, w, h, fz+fd-(pz+d)) if pz+d < fz+fd
            
            # Actually, to maintain maximality and disjointness, we usually generate sub-spaces
            # that are disjoint from the placed box and each other.
            
            # Part 1: x < px
            if px > fx:
                new_spaces.append((fx, fy, fz, px - fx, fh, fd))
            
            # Part 2: x > px + w
            if px + w < fx + fw:
                new_spaces.append((px + w, fy, fz, fx + fw - (px + w), fh, fd))
            
            # Part 3: y < py (and x in [px, px+w])
            if py > fy:
                new_spaces.append((px, fy, fz, w, py - fy, d))
            
            # Part 4: y > py + h (and x in [px, px+w])
            if py + h < fy + fh:
                new_spaces.append((px, py + h, fz, w, fy + fh - (py + h), d))
            
            # Part 5: z < pz (and x in [px, px+w], y in [py, py+h])
            if pz > fz:
                new_spaces.append((px, py, fz, w, h, pz - fz))
            
            # Part 6: z > pz + d (and x in [px, px+w], y in [py, py+h])
            if pz + d < fz + fd:
                new_spaces.append((px, py, pz + d, w, h, fz + fd - (pz + d)))
                
        return new_spaces

    # We need a more efficient way to find the best position.
    # For each bin, we iterate through its free spaces.
    # For each free space, the only valid positions to place the box without creating non-maximal spaces
    # or overlapping are the "bottom-left-front" corners relative to that free space?
    # No, if we place a box in a free space, we can place it anywhere within that free space.
    # However, to maintain the property that we only need to check against free spaces,
    # we typically assume we place the box at the "lowest" possible position in that free space
    # to minimize fragmentation? 
    # Actually, the standard "Maximal Space" approach (like in some 3D bin packing heuristics)
    # allows placing the box at ANY position within a free space. 
    # But checking all positions is infinite. 
    # The key insight is: if a box fits in a free space, it fits at the corner (fx, fy, fz).
    # Why? Because if it fits at (px, py, pz), it also fits at (fx, fy, fz) since the free space
    # is empty. Placing it at (fx, fy, fz) is generally better for packing (keeps higher space free).
    # So, for each free space, we only consider placing the box at the corner (fx, fy, fz).
    
    # Algorithm:
    # 1. Sort boxes by volume desc.
    # 2. For each box:
    #    a. Find the best bin and best free space within that bin.
    #       "Best" is defined by:
    #       - Primary: Minimize the remaining volume in the free space after placement? 
    #         Or minimize the "waste" in the specific dimensions?
    #         Common heuristic: Best Fit = place in the space where the remaining space is smallest.
    #         Alternatively, First Fit Decreasing with Best Fit within bin.
    #       Let's use: For each bin, find the free space that minimizes the "score" of the placement.
    #       Score could be: volume of remaining space in that free space? 
    #       Or: sum of (fw-w)^2 + (fh-h)^2 + (fd-d)^2? 
    #       Let's try: Minimize the volume of the free space after splitting?
    #       Actually, a simpler and effective heuristic is:
    #       For each bin, check all free spaces. If box fits at (fx, fy, fz), calculate a "fit score".
    #       Pick the bin/space with the best (lowest) score.
    #       If no existing bin works, create a new bin.
    
    # Let's refine the "best fit" metric.
    # A good metric is the "tightness" of the fit.
    # We want to fill the space as completely as possible.
    # So, for a free space (fw, fh, fd) and box (w, h, d):
    # Waste = (fw-w)*(fh-h)*fd + fw*(fh-h)*(fd-d) + fw*fh*(fd-d) ... this is complex.
    # Simpler: Minimize the volume of the largest remaining sub-space?
    # Or simply: Minimize the volume of the free space itself? (Smallest space that fits)
    # This is "Best Fit" in 1D. Let's try that: pick the smallest free space that fits the box.
    
    # However, we also need to consider which BIN to pick.
    # BFD usually picks the first bin that fits? Or the best bin?
    # "Best-Fit Decreasing" usually implies:
    # 1. Sort items decreasing.
    # 2. For each item, find the bin where it fits in the "best" way.
    #    "Best" often means: the bin where the item causes the least remaining empty space,
    #    or the item is placed in the "tightest" spot.
    
    # Let's implement:
    # For each box:
    #   best_bin_idx = -1
    #   best_space_idx = -1
    #   best_score = infinity
    #   for each bin b:
    #       for each space s in b:
    #           if box fits in s:
    #               score = calculate_score(s, box)
    #               if score < best_score:
    #                   best_score = score
    #                   best_bin_idx = b
    #                   best_space_idx = s
    #   if best_bin_idx == -1:
    #       create new bin, place in it
    #   else:
    #       place in best_bin_idx at best_space_idx
    
    # What score?
    # Let's use: Volume of the free space. Smaller free space is better (tighter fit).
    # This encourages filling small gaps.
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin or just append to last bin (might overlap, but better than crash? No, must be valid)
            # If we run out of time, we must return a valid solution.
            # We have placed some boxes. We need to place the rest.
            # We can just put all remaining boxes in a single new bin, but they might overlap.
            # To ensure validity, we can use a simple shelf method for the remainder.
            # Or, since we have time guard, we should have finished.
            # If we hit deadline, we return what we have? No, must place ALL boxes.
            # Let's assume we don't hit deadline for 40 boxes.
            # But to be safe:
            # If deadline exceeded, use a fast fallback for remaining boxes.
            pass
            
        best_bin_idx = -1
        best_space_idx = -1
        best_score = float('inf')
        
        for b_idx, free_spaces in enumerate(bins):
            for s_idx, (fx, fy, fz, fw, fh, fd) in enumerate(free_spaces):
                if w <= fw and h <= fh and d <= fd:
                    # Calculate score. 
                    # Smaller free space volume is better.
                    score = fw * fh * fd
                    # Tie-breaker: prefer lower z (bottom), then y, then x?
                    # If scores are equal, prefer the one that places the box lower?
                    # Let's add a small penalty for height to encourage stacking.
                    # score = fw*fh*fd + 0.001 * fz
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_space_idx = s_idx
        
        if best_bin_idx == -1:
            # No existing bin fits. Create new bin.
            bins.append(new_bin())
            best_bin_idx = len(bins) - 1
            best_space_idx = 0 # The only space is the whole container
            
        # Place the box in the selected space
        # Get the space
        fx, fy, fz, fw, fh, fd = bins[best_bin_idx][best_space_idx]
        
        # Place at (fx, fy, fz)
        px, py, pz = fx, fy, fz
        
        # Record placement
        placements[idx] = (best_bin_idx, px, py, pz, w, h, d)
        
        # Update free spaces in the bin
        # Remove the old space
        old_space = bins[best_bin_idx].pop(best_space_idx)
        
        # Generate new spaces
        new_spaces = split_space(old_space[0], old_space[1], old_space[2], old_space[3], old_space[4], old_space[5], px, py, pz, w, h, d)
        
        # Add new spaces to the bin's list
        bins[best_bin_idx].extend(new_spaces)
        
        # Optional: Merge adjacent free spaces? 
        # This is complex and might be slow. For 40 boxes, the number of spaces might grow, but not exponentially if we are careful.
        # However, without merging, we might have many small, fragmented spaces.
        # The "Best Fit" (smallest space) heuristic will tend to use small spaces, which is good.
        
    # The bin indices in placements are 0-indexed in our internal list.
    # The problem says "Bin labels are nonnegative integers". 0 is fine.
    # "The largest label defines the last bin".
    # We return the list of placements in the original order of boxes.
    
    return placements
