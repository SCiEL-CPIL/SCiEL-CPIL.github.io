# EVOLVE-BLOCK-START
import numpy as np
import time

def find_best_placement(grid, w, h, d, W, H, D):
    """
    Finds the best placement for a box of size (w, h, d) in a grid of size (W, H, D).
    'Best' is defined as the position that minimizes the remaining empty volume in the bin
    (i.e., maximizes the filled volume), subject to the box fitting.
    If multiple positions have the same waste, we prefer the one with smallest z, then x, then y.
    Returns (x, y, z) or None if no placement is found.
    """
    if w > W or h > H or d > D:
        return None
    
    best_pos = None
    min_waste = float('inf')
    
    # Iterate over possible z positions (height)
    # We prioritize lower z to keep the stack compact, but best-fit might override this
    # if a higher z allows a tighter fit in x/y.
    # However, standard best-fit usually minimizes wasted space.
    # Let's iterate z from 0 to D-d.
    for z in range(D - d + 1):
        # Sum occupancy in the z-slice range [z, z+d)
        # grid shape is (W, H, D)
        # slice is grid[:, :, z:z+d]
        if d == 1:
            slice_occupancy = grid[:, :, z]
        else:
            # If d > 1, we need to check if ANY voxel in the z-range is occupied.
            # Summing along axis 2 gives the count of occupied voxels in that column (x,y) across the z-depth.
            # If sum > 0, the box cannot be placed at (x,y) for this z-range.
            slice_occupancy = np.sum(grid[:, :, z:z+d], axis=2)
        
        # We need to find (x, y) such that slice_occupancy[x:x+w, y:y+h] is all zeros.
        # This is equivalent to the sum of that window being 0.
        
        # Use sliding_window_view to get all (w, h) windows
        # Shape of slice_occupancy is (W, H)
        # Shape of windows will be (W-w+1, H-h+1, w, h)
        if W - w + 1 <= 0 or H - h + 1 <= 0:
            continue
            
        try:
            windows = np.lib.stride_tricks.sliding_window_view(slice_occupancy, (w, h))
            # Sum over the last two axes (the window dimensions)
            window_sums = np.sum(windows, axis=(2, 3))
            
            # Valid positions are where window_sums == 0
            valid = window_sums == 0
            
            if not np.any(valid):
                continue
                
            # For all valid positions, calculate the "waste" or "impact".
            # A simple proxy for waste is the sum of the surrounding free space?
            # Actually, the total volume of the bin is fixed.
            # The "waste" created by placing a box is not just the box volume, but how it fragments the space.
            # A common heuristic is to minimize the "bounding box" of the placed items or maximize the "connected free space" at the bottom.
            # However, a simpler and effective heuristic for 3D BFD is to minimize the z-coordinate (put things low) 
            # and among same z, minimize x, then y. This is essentially "First Fit" with a specific order.
            
            # Let's try a slightly smarter metric: 
            # Minimize the "distance" from the origin (0,0,0) for the box's center of mass?
            # Or simply: among valid positions, pick the one with smallest z.
            # If multiple have same z, pick smallest x.
            # If multiple have same x, pick smallest y.
            # This is effectively the "First Fit" in z, then x, then y order.
            # Is this better than pure First Fit? 
            # Pure First Fit in the current code iterates bins, then z, then x, then y.
            # So for a single bin, it already picks the "lowest, leftmost, rearmost" position.
            # The improvement comes from choosing the BEST BIN, not just the first one.
            # So we need a metric to compare bins.
            
            # Let's use the occupied volume of the bin after placement as a tie-breaker?
            # No, we want to fill bins up.
            # Let's calculate the "waste" as the total free volume in the bin.
            # Since we are adding the same box volume to all candidates, the bin with the LARGEST current occupied volume
            # that can fit the box will have the SMALLEST remaining free volume.
            # So, "Best Fit" means placing the box in the bin that is ALREADY MOST FULLY OCCUPIED (among those that can fit it).
            
            # Therefore, in the main loop, we should track the occupied volume of each bin.
            # We don't need a complex per-position waste calculation if we assume that filling the fullest bin is best.
            # This is the standard "Best Fit Decreasing" heuristic.
            
            # So, find the first valid position in this bin (lowest z, then x, then y) to return.
            # The main loop will handle the bin selection based on bin occupancy.
            
            # Find the first True in row-major order (z is fixed in this loop, so x then y)
            # np.argwhere returns array of [x, y]
            coords = np.argwhere(valid)
            if len(coords) > 0:
                x, y = coords[0]
                # We found a valid position in this bin at this z.
                # Since we iterate z from 0 upwards, this is the lowest z.
                # Since we take the first (x,y) in row-major, it's the smallest x, then smallest y.
                return (x, y, z)
                
        except Exception:
            continue
            
    return None

def pack(boxes, container):
    W, H, D = container
    n_boxes = len(boxes)
    
    # Sort boxes by volume descending
    box_volumes = [w*h*d for w,h,d in boxes]
    sorted_indices = np.argsort(box_volumes)[::-1]
    
    placements = [None] * n_boxes
    bins = [] # List of dicts: {'grid': np.array, 'vol': int}
    
    for orig_idx in sorted_indices:
        w, h, d = boxes[orig_idx]
        box_vol = w * h * d
        placed = False
        
        # Best Fit Decreasing:
        # 1. Find all bins where the box fits (volume-wise and dimension-wise).
        # 2. Among those, pick the bin with the HIGHEST current occupied volume.
        # 3. Place the box in that bin at the best position (lowest z, then x, then y).
        
        best_bin_idx = -1
        best_bin_vol = -1
        best_pos = None
        
        for bin_idx, bin_data in enumerate(bins):
            # Check volume fit
            if bin_data['vol'] + box_vol > W*H*D:
                continue
            
            # Check dimension fit (quick check)
            if w > W or h > H or d > D:
                continue
                
            # Try to find a position in this bin
            grid = bin_data['grid']
            pos = find_best_placement(grid, w, h, d, W, H, D)
            
            if pos is not None:
                # This bin can fit the box.
                # Is it a better fit than the current best?
                # "Better" means higher current volume (more full).
                if bin_data['vol'] > best_bin_vol:
                    best_bin_vol = bin_data['vol']
                    best_bin_idx = bin_idx
                    best_pos = pos
                    # Optimization: If this bin is almost full (e.g., > 90%), we can't get much better.
                    # But checking all is safer for correctness and small N.
        
        if best_bin_idx != -1:
            # Place in the best bin
            x, y, z = best_pos
            bin_data = bins[best_bin_idx]
            grid = bin_data['grid']
            grid[x:x+w, y:y+h, z:z+d] = 1
            bin_data['vol'] += box_vol
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
            placed = True
        else:
            # No existing bin fits. Create a new one.
            grid = np.zeros((W, H, D), dtype=np.uint8)
            grid[0:w, 0:h, 0:d] = 1
            bins.append({'grid': grid, 'vol': box_vol})
            placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
            placed = True
            
    return placements

# EVOLVE-BLOCK-END
