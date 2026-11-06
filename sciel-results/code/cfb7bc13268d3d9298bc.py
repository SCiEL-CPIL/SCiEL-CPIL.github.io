<<<<<<< SEARCH
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
=======
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s kill
    container_vol = W * H * D

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    remaining_vols = [0.0] * n
>>>>>>> REPLACE
<<<<<<< SEARCH
    def find_position(bin_placements, w, h, d):
        """
        Try multiple placement strategies and return the best position.
        """
        # Try corner-based
        pos_corner = find_position_corner(bin_placements, w, h, d)
        if pos_corner is not None:
            return pos_corner
        
        # Try shelf-based
        pos_shelf = find_position_shelf(bin_placements, w, h, d)
        if pos_shelf is not None:
            return pos_shelf
        
        return None
=======
    def find_position(bin_placements, w, h, d):
        """
        Try multiple placement strategies and return the best position.
        """
        # Try corner-based
        pos_corner = find_position_corner(bin_placements, w, h, d)
        if pos_corner is not None:
            return pos_corner
        
        # Try shelf-based
        pos_shelf = find_position_shelf(bin_placements, w, h, d)
        if pos_shelf is not None:
            return pos_shelf
        
        return None

    def bin_fill_ratio(bin_placements):
        """Calculate the volume fill ratio of a bin."""
        if not bin_placements:
            return 0.0
        vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_placements)
        return vol / container_vol if container_vol > 0 else 0.0
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Main packing loop
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins, preferring the bin where it fits tightest
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize the resulting bounding box
                fit_score = (x + w, y + h, z + d)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_bin = []
            x, y, z = 0, 0, 0
            if can_place(new_bin, x, y, z, w, h, d):
                new_bin.append((x, y, z, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, x, y, z, w, h, d)
            else:
                # This shouldn't happen if container is large enough, but handle gracefully
                new_bin.append((0, 0, 0, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
=======
    # Main packing loop
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins, preferring the bin where it fits tightest
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize the resulting bounding box
                fit_score = (x + w, y + h, z + d)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            result[idx] = (best_bin, x, y, z, w, h, d)
            remaining_vols[idx] = vol
        else:
            # Create a new bin
            new_bin = []
            x, y, z = 0, 0, 0
            if can_place(new_bin, x, y, z, w, h, d):
                new_bin.append((x, y, z, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, x, y, z, w, h, d)
                remaining_vols[idx] = vol
            else:
                # This shouldn't happen if container is large enough, but handle gracefully
                new_bin.append((0, 0, 0, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
                remaining_vols[idx] = vol
>>>>>>> REPLACE
<<<<<<< SEARCH
    # If we hit the deadline, place remaining boxes using a simple fallback
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        # Find any bin where it can be placed, or create a new one
        placed = False
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    return result
=======
    # If we hit the deadline, place remaining boxes using a simple fallback
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        # Find any bin where it can be placed, or create a new one
        placed = False
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Improvement pass: try to move boxes to reduce bin count or improve last bin fill
    # We can try to remove boxes from the last bin and place them in earlier bins
    # Or try to consolidate bins
    
    # First, let's identify the "last bin" (highest index)
    if bins:
        last_bin_idx = max(b_idx for b_idx, placements in enumerate(bins) if placements)
        
        # Try to move boxes from last bin to other bins
        # We want to empty the last bin if possible, or at least reduce its volume
        # Sort boxes in last bin by volume descending to try moving large ones first
        last_bin_boxes = [(i, w, h, d, x, y, z) for i, (w, h, d, x, y, z) in 
                          [(result[i][1], boxes[i][0], boxes[i][1], boxes[i][2], result[i][1], result[i][2], result[i][3]) 
                           for i in range(n) if result[i] and result[i][0] == last_bin_idx]]
        
        # Actually, let's just iterate through all boxes in the last bin
        boxes_in_last_bin = [i for i in range(n) if result[i] and result[i][0] == last_bin_idx]
        # Sort by volume descending
        boxes_in_last_bin.sort(key=lambda i: box_vols[i], reverse=True)
        
        moved_any = False
        for i in boxes_in_last_bin:
            if time.time() > deadline:
                break
            # Try to remove box i from last bin and place it in another bin
            w, h, d = boxes[i]
            # Temporarily remove from last bin
            old_pos = result[i]
            # Find the box in last_bin
            box_to_remove = None
            for j, (px, py, pz, pw, ph, pd) in enumerate(bins[last_bin_idx]):
                if (pw, ph, pd) == (w, h, d):
                    box_to_remove = j
                    break
            
            if box_to_remove is None:
                continue
                
            # Remove it
            bins[last_bin_idx].pop(box_to_remove)
            
            # Try to place in other bins
            placed = False
            for b_idx in range(len(bins)):
                if b_idx == last_bin_idx:
                    continue
                pos = find_position(bins[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    result[i] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    moved_any = True
                    break
            
            if not placed:
                # Put it back
                bins[last_bin_idx].append((old_pos[1], old_pos[2], old_pos[3], w, h, d))
                # Restore original position
                result[i] = (last_bin_idx, old_pos[1], old_pos[2], old_pos[3], w, h, d)
        
        # If we emptied the last bin, remove it
        if bins and not bins[last_bin_idx]:
            bins.pop(last_bin_idx)
            # Update bin indices for all results
            for i in range(n):
                if result[i] and result[i][0] > last_bin_idx:
                    result[i] = (result[i][0] - 1, result[i][1], result[i][2], result[i][3], 
                                result[i][4], result[i][5], result[i][6])
                elif result[i] and result[i][0] == last_bin_idx:
                    # This shouldn't happen if we emptied it
                    pass

    return result
>>>>>>> REPLACE
