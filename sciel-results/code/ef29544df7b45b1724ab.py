import time
import itertools

def load(boxes, container):
    """
    Loads boxes into a container using a 3D Extent Tree (Skyline) approach.
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit
    
    W, H, D = int(container[0]), int(container[1]), int(container[2])
    n = len(boxes)
    
    # Fallback for extremely small containers or if something goes wrong
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    # Order boxes by volume (largest first) to prioritize big items
    # We keep the original index to return results in the correct order
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # Extents: List of tuples (x, y, z, w, h, d) representing free rectangular prisms
    # Initially, the whole container is free
    extents = [(0, 0, 0, W, H, D)]
    
    placements = [None] * n
    placed_count = 0
    unplaced_indices = []

    def get_orientations(w, h, d):
        """Yield unique permutations of dimensions."""
        seen = set()
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                yield perm

    def split_extent(ext, box_w, box_h, box_d):
        """
        Given an extent (x, y, z, w, h, d) and a box placed at the bottom-left-front (x,y,z) 
        with dimensions (box_w, box_h, box_d), split the extent into new free extents.
        The box is placed at the 'origin' of the extent (x, y, z).
        """
        x, y, z, w, h, d = ext
        new_extents = []
        
        # 1. Split along X axis (if there is space to the right)
        if box_w < w:
            new_extents.append((x + box_w, y, z, w - box_w, h, d))
            
        # 2. Split along Y axis (if there is space above)
        # Note: The region above the box is only valid for the full width 'w' 
        # if we consider the L-shape decomposition properly.
        # Standard L-shape decomposition:
        # Region A: To the right of the box (x+box_w, y, z, w-box_w, h, d)
        # Region B: Above the box, but only over the width of the box? 
        # Actually, a simpler decomposition is:
        # The remaining space is:
        # 1. (x + box_w, y, z, w - box_w, h, d)  [Right]
        # 2. (x, y + box_h, z, w, h - box_h, d) [Top]
        # 3. (x, y, z + box_d, w, h, d - box_d) [Back]
        # Wait, this overlaps. 
        # Correct L-shape (3 boxes) decomposition of a rectangular prism with a corner box removed:
        # Let box be at (x,y,z) with size (bw, bh, bd).
        # Extent is (x,y,z,w,h,d).
        
        # Piece 1: To the right of the box
        if box_w < w:
            new_extents.append((x + box_w, y, z, w - box_w, h, d))
            
        # Piece 2: Above the box, but only in the 'left' part (width bw)
        # Why only left? Because the 'right' part is already covered by Piece 1's vertical extension?
        # No, Piece 1 is (x+bw, y, z, w-bw, h, d). It covers the right side fully in height and depth.
        # So the remaining space above the box is only over the width 'bw'.
        if box_h < h:
            new_extents.append((x, y + box_h, z, box_w, h - box_h, d))
            
        # Piece 3: Behind the box, but only in the 'bottom-left' part (width bw, height bh)
        # Why? Because:
        # - The right part (width w-bw) is covered by Piece 1.
        # - The top part (height h-bh) over the left width (bw) is covered by Piece 2.
        # So the remaining space behind is only in the region (x, y, z+bd) with size (bw, bh, d-bd).
        if box_d < d:
            new_extents.append((x, y, z + box_d, box_w, box_h, d - box_d))

        return new_extents

    # Process boxes in sorted order
    for bi in order:
        if time.time() > deadline:
            unplaced_indices.append(bi)
            continue
            
        w, h, d = boxes[bi]
        best_fit = None # (score, extent_index, orientation)
        
        # Iterate through all current free extents
        # We need to be careful: if we place a box, we modify the extents list.
        # So we should find the best fit first, then apply it.
        
        for ei, ext in enumerate(extents):
            ex, ey, ez, ew, eh, ed = ext
            
            # Check if box fits in this extent for any orientation
            for ow, oh, od in get_orientations(w, h, d):
                if ow <= ew and oh <= eh and od <= ed:
                    # It fits. Calculate score.
                    # We prefer packing low (z), then low y, then low x.
                    # Since the box is placed at the bottom-left-front of the extent (ex, ey, ez),
                    # the position is fixed relative to the extent.
                    score = (ez, ey, ex)
                    # Tie-breaker: prefer larger remaining space? Or just keep first found?
                    # Let's stick to bottom-left-front preference.
                    if best_fit is None or score < best_fit[0]:
                        best_fit = (score, ei, (ow, oh, od))
        
        if best_fit is not None:
            _, best_ei, (ow, oh, od) = best_fit
            ex, ey, ez, ew, eh, ed = extents[best_ei]
            
            # Place box at (ex, ey, ez)
            placements[bi] = (0, ex, ey, ez, ow, oh, od)
            placed_count += 1
            
            # Update extents: remove the one we used and add the new ones
            new_extents = split_extent(extents[best_ei], ow, oh, od)
            extents[best_ei] = new_extents[0] if new_extents else None
            if len(new_extents) > 1:
                # Replace the single extent with multiple ones
                # This modifies the list structure, so we need to handle indices carefully in the next iteration
                # Actually, since we just found the best fit and are now committing, 
                # we can just update the list for the next box.
                extents = [e for e in extents if e is not None]
                extents.extend(new_extents[1:])
                # Wait, the logic above is messy. 
                # Better: 
                # extents[best_ei] was the one used.
                # We replace it with new_extents.
                # Since new_extents can be 0, 1, 2, or 3 items, we need to splice the list.
                
                # Let's restart the extent update logic properly
                pass # Placeholder for correct logic below
        else:
            unplaced_indices.append(bi)
            
    # The extent update logic inside the loop was flawed because modifying the list while iterating or referencing indices is tricky if the list size changes.
    # Let's re-implement the main loop with a cleaner extent management strategy.
    
    # --- RE-IMPLEMENTATION WITH CLEANER EXTENT MANAGEMENT ---
    
    extents = [(0, 0, 0, W, H, D)]
    placements = [None] * n
    placed_count = 0
    unplaced_indices = []

    for bi in order:
        if time.time() > deadline:
            unplaced_indices.append(bi)
            continue
            
        w, h, d = boxes[bi]
        best_fit = None # (score, extent_index, orientation)
        
        for ei, ext in enumerate(extents):
            ex, ey, ez, ew, eh, ed = ext
            for ow, oh, od in get_orientations(w, h, d):
                if ow <= ew and oh <= eh and od <= ed:
                    score = (ez, ey, ex)
                    if best_fit is None or score < best_fit[0]:
                        best_fit = (score, ei, (ow, oh, od))
        
        if best_fit is not None:
            _, best_ei, (ow, oh, od) = best_fit
            ex, ey, ez, ew, eh, ed = extents[best_ei]
            
            placements[bi] = (0, ex, ey, ez, ow, oh, od)
            placed_count += 1
            
            # Generate new extents from the split
            new_extents = split_extent(extents[best_ei], ow, oh, od)
            
            # Replace the used extent with the new ones
            # Remove the old one
            extents.pop(best_ei)
            # Add the new ones
            extents.extend(new_extents)
        else:
            unplaced_indices.append(bi)
            
    # Mark unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements

