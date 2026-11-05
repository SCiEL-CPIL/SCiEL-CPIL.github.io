"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # We'll use a list of bins, each bin is a list of (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if box at (x,y,z) with dims (w,h,d) fits in the bin without overlap."""
        for px, py, pz, pw, ph, pd in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_position(bin_placements, w, h, d):
        """Find the best (bottom-most, then y, then x) position for a box in a bin."""
        # Generate candidate positions
        # Start from (0,0,0) and try all combinations of existing box edges
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for px, py, pz, pw, ph, pd in bin_placements:
            if px + pw <= W:
                x_candidates.append(px + pw)
            if py + ph <= H:
                y_candidates.append(py + ph)
            if pz + pd <= D:
                z_candidates.append(pz + pd)
        
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        best_pos = None
        best_score = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_volume(bin_placements):
        return sum(w*h*d for _, _, _, w, h, d in bin_placements)
    
    # Place boxes one by one
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
        
        placed = False
        
        # Try to place in existing bins first (best fit)
        best_bin = -1
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                # Best fit: minimize the "wasted space" - prefer bins where the box fits tightly
                # Use the position as a proxy: lower z is better
                score = pos[2]  # z-coordinate
                if best_fit_score is None or score < best_fit_score:
                    best_fit_score = score
                    best_bin = b_idx
        
        if best_bin >= 0:
            x, y, z = find_position(bins[best_bin], w, h, d)
            bins[best_bin].append((x, y, z, w, h, d))
            placed = True
        else:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
            placed = True
        
        if not placed:
            # Should not happen, but fallback
            bins.append([(0, 0, 0, w, h, d)])
    
    # Build the result
    result = [None] * n
    for b_idx, bin_placements in enumerate(bins):
        for x, y, z, w, h, d in bin_placements:
            # Find which original box this is
            # We need to track this properly
            pass
    
    # Let me redo this with proper tracking
    # Actually, let me restructure to track original indices
    
    # Reset and redo with proper index tracking
    bins = []  # Each bin: list of (orig_index, x, y, z, w, h, d)
    
    def find_position_v2(bin_placements, w, h, d):
        """Find best position, ignoring orig_index in overlap check."""
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for _, px, py, pz, pw, ph, pd in bin_placements:
            if px + pw <= W:
                x_candidates.append(px + pw)
            if py + ph <= H:
                y_candidates.append(py + ph)
            if pz + pd <= D:
                z_candidates.append(pz + pd)
        
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        best_pos = None
        best_score = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    # Check overlap
                    overlap = False
                    for _, px, py, pz, pw, ph, pd in bin_placements:
                        if (x < px + pw and x + w > px and
                            y < py + ph and y + h > py and
                            z < pz + pd and z + d > pz):
                            overlap = True
                            break
                    if not overlap:
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position_v2(bin_placements, w, h, d)
            if pos is not None:
                score = pos[2]
                if best_fit_score is None or score < best_fit_score:
                    best_fit_score = score
                    best_bin = b_idx
        
        if best_bin >= 0:
            x, y, z = find_position_v2(bins[best_bin], w, h, d)
            bins[best_bin].append((orig_i, x, y, z, w, h, d))
        else:
            bins.append([(orig_i, 0, 0, 0, w, h, d)])
    
    # Build result
    result = [None] * n
    for b_idx, bin_placements in enumerate(bins):
        for orig_i, x, y, z, w, h, d in bin_placements:
            result[orig_i] = (b_idx, x, y, z, w, h, d)
    
    # Fallback for any unplaced boxes (shouldn't happen)
    next_bin = len(bins)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (next_bin, 0, 0, 0, w, h, d)
            next_bin += 1
    
    return result
# EVOLVE-BLOCK-END
