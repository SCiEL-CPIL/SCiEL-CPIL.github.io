import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Define sorting strategies
    # We will try a few different orderings and pick the best result
    strategies = [
        lambda b: b[0]*b[1]*b[2], # Volume
        lambda b: max(b),          # Max dimension
        lambda b: b[0]*b[1] + b[1]*b[2] + b[0]*b[2], # Surface area
        lambda b: b[0]*b[1]*b[2] / max(b), # Volume / max dim (compactness)
    ]
    
    best_placements = None
    best_score = float('inf')
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        # Sort boxes
        sorted_indices = sorted(range(n), key=lambda i: -strategy(boxes[i]))
        sorted_boxes = [boxes[i] for i in sorted_indices]
        
        # Run BFD/FFD
        placements = _run_bfd(sorted_boxes, W, H, D)
        
        # Calculate score
        # B = number of occupied bins
        # Last bin occupied volume fraction
        if not placements:
            continue
            
        bin_ids = [p[0] for p in placements]
        B = max(bin_ids) + 1 if bin_ids else 0
        
        # Calculate volume of last bin
        last_bin_id = max(bin_ids)
        last_bin_vol = 0
        container_vol = W * H * D
        for p in placements:
            if p[0] == last_bin_id:
                last_bin_vol += p[4] * p[5] * p[6]
                
        fill_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - fill_fraction)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            # Map back to original indices? 
            # The task requires: "one placement (bin, x, y, z, w, h, d) per box, preserving its original ordered dimensions"
            # The return value is a list of placements. Does the order of the list matter?
            # "preserving its original ordered dimensions" refers to w,h,d not being rotated.
            # The output list usually corresponds to the input box order?
            # "places every box... into numbered bins... scored by... on five generated 40-box instances"
            # The interface is pack(boxes, container) -> one placement per box.
            # It doesn't explicitly say the output list must be in the same order as input boxes, 
            # but "preserving its original ordered dimensions" suggests we shouldn't rotate.
            # However, if the evaluator checks `placements[i]` corresponds to `boxes[i]`, we must map back.
            # Let's assume the output list index i corresponds to input box i.
            
    # If we didn't map back in the loop, we need to handle the mapping.
    # Let's refactor _run_bfd to return the placement for each *original* box index.
    
    # Re-run with mapping for the best strategy? 
    # It's better to do it inside the loop.
    
    # Let's rewrite the loop to track the best *strategy* and then run it once with mapping, 
    # or just keep the mapped placements.
    
    # I will modify _run_bfd to accept original indices and return placements in original order.
    
    best_placements_mapped = None
    best_score = float('inf')
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sorted(range(n), key=lambda i: -strategy(boxes[i]))
        
        # _run_bfd_mapped returns a list of placements where result[i] is the placement for boxes[i]
        placements_mapped = _run_bfd_mapped(boxes, sorted_indices, W, H, D)
        
        bin_ids = [p[0] for p in placements_mapped]
        B = max(bin_ids) + 1 if bin_ids else 0
        
        if B == 0:
            continue
            
        last_bin_id = max(bin_ids)
        last_bin_vol = 0
        container_vol = W * H * D
        for p in placements_mapped:
            if p[0] == last_bin_id:
                last_bin_vol += p[4] * p[5] * p[6]
                
        fill_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - fill_fraction)
        
        if score < best_score:
            best_score = score
            best_placements_mapped = placements_mapped
            
    if best_placements_mapped is None:
        # Fallback: simple shelf
        return _shelf_fallback(boxes, W, H, D)
        
    return best_placements_mapped

def _run_bfd_mapped(boxes, sorted_indices, W, H, D):
    n = len(boxes)
    # bins: list of lists of free spaces
    # free space: (x, y, z, w, h, d)
    bins = []
    # placement_result[i] = (bin_id, x, y, z, w, h, d)
    placement_result = [None] * n
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        # Try existing bins
        for b_idx, free_spaces in enumerate(bins):
            # Check all free spaces in this bin
            for fs_idx, fs in enumerate(free_spaces):
                fx, fy, fz, fw, fh, fd = fs
                if w <= fw and h <= fh and d <= fd:
                    # It fits. Calculate a "waste" metric to choose best position in this bin.
                    # We prefer positions that leave larger rectangular spaces.
                    # Simple metric: sum of dimensions of resulting spaces?
                    # Or just take the first fit in the bin (First Fit).
                    # Let's try minimizing the "corner" distance or similar.
                    # For now, let's just take the first fit in the first bin that works (First Fit Decreasing).
                    # This is fast and often good.
                    best_bin = b_idx
                    best_pos = (fx, fy, fz)
                    best_waste = 0 # Break immediately? 
                    # To improve, we could scan all free spaces in this bin and pick the "best" one.
                    # "Best" could be the one that minimizes the remaining volume of the free space.
                    rem_vol = fw*fh*fd - w*h*d
                    if rem_vol < best_waste:
                        best_waste = rem_vol
                        best_pos = (fx, fy, fz)
                        # Note: we are still in the same bin b_idx. 
                        # We should finish checking all free spaces in this bin to find the best one in THIS bin.
                        # But do we want to check other bins? 
                        # BFD: Best Fit Decreasing. 
                        # Usually, we pick the bin that minimizes the wasted space in the bin? 
                        # Or just the first bin? 
                        # Let's stick to First Fit for bins (lowest index) but Best Fit for position within that bin.
                        # This is a hybrid. Let's call it FFD with local best position.
                    break # Stop checking other free spaces in this bin? 
                    # No, we want the best position in this bin. So we should NOT break the inner loop immediately if we want best position.
                    # But if we break the outer loop (bins) immediately, we are doing First Fit for bins.
                    # Let's do: 
                    # 1. Iterate bins. 
                    # 2. For the first bin that has ANY fit, find the BEST fit in that bin. 
                    # 3. Place it there. 
                    # This is efficient.
                    pass # Logic below
        
        # Corrected Logic for FFD with Best Position in First Fit Bin:
        placed = False
        for b_idx, free_spaces in enumerate(bins):
            best_fs_idx_in_bin = -1
            best_rem_vol = float('inf')
            
            for fs_idx, fs in enumerate(free_spaces):
                fx, fy, fz, fw, fh, fd = fs
                if w <= fw and h <= fh and d <= fd:
                    rem_vol = (fw*fh*fd) - (w*h*d)
                    if rem_vol < best_rem_vol:
                        best_rem_vol = rem_vol
                        best_fs_idx_in_bin = fs_idx
            
            if best_fs_idx_in_bin != -1:
                # We found a fit in this bin. Place it in the best position.
                fx, fy, fz, fw, fh, fd = free_spaces[best_fs_idx_in_bin]
                
                # Update free spaces
                new_spaces = []
                
                # 1. Back space
                if fd > d:
                    new_spaces.append((fx, fy, fz + d, fw, fh, fd - d))
                
                # 2. Right space (in the front slice)
                if fw > w:
                    new_spaces.append((fx + w, fy, fz, fw - w, fh, d))
                
                # 3. Top space (in the front-right slice)
                if fh > h:
                    new_spaces.append((fx, fy + h, fz, w, fh - h, d))
                
                # Replace the used free space
                free_spaces[best_fs_idx_in_bin] = None # Mark for removal
                # Actually, easier to rebuild the list
                bins[b_idx] = [fs for fs in free_spaces if fs is not None] + new_spaces
                
                placement_result[idx] = (b_idx, fx, fy, fz, w, h, d)
                placed = True
                break # Move to next box
        
        if not placed:
            # Open new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            
            # Place in the only free space
            fx, fy, fz, fw, fh, fd = 0, 0, 0, W, H, D
            new_spaces = []
            if fd > d:
                new_spaces.append((fx, fy, fz + d, fw, fh, fd - d))
            if fw > w:
                new_spaces.append((fx + w, fy, fz, fw - w, fh, d))
            if fh > h:
                new_spaces.append((fx, fy + h, fz, w, fh - h, d))
            
            bins[b_idx] = new_spaces
            placement_result[idx] = (b_idx, fx, fy, fz, w, h, d)
            
    return placement_result

def _shelf_fallback(boxes, W, H, D):
    placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements

