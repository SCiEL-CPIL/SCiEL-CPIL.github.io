"""3D Bin Packing: place boxes unrotated into numbered bins, minimize bins + last-bin fill penalty."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    3D Bin Packing with Best-Fit Decreasing using occupancy grids for fast collision checks.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort boxes by volume descending, keep original index
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[1])
    
    # Data structures
    bins = []  # list of dicts: {'boxes': [(x,y,z,w,h,d)], 'grid': 3D bool array, 'volume': int}
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def create_bin():
        """Create a new empty bin with a 3D occupancy grid."""
        return {
            'boxes': [],
            'grid': np.zeros((W, H, D), dtype=bool),
            'volume': 0
        }
    
    def can_place_grid(grid, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) using grid slice."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any cell in the box's region is occupied
        return not grid[x:x+w, y:y+h, z:z+d].any()
    
    def place_in_bin(bin_data, x, y, z, w, h, d):
        """Mark cells as occupied in the bin's grid."""
        bin_data['grid'][x:x+w, y:y+h, z:z+d] = True
        bin_data['boxes'].append((x, y, z, w, h, d))
        bin_data['volume'] += w * h * d
    
    def unplace_in_bin(bin_data, x, y, z, w, h, d):
        """Mark cells as unoccupied in the bin's grid."""
        bin_data['grid'][x:x+w, y:y+h, z:z+d] = False
        # Remove the box from the list
        bin_data['boxes'] = [b for b in bin_data['boxes'] 
                            if not (b[0] == x and b[1] == y and b[2] == z)]
        bin_data['volume'] -= w * h * d
    
    def find_best_spot(bin_data, w, h, d):
        """
        Find the best (x, y, z) position for a box in a given bin using grid checks.
        Uses candidate positions derived from box faces.
        Returns (x, y, z) or None if no valid position found.
        """
        if not bin_data['boxes']:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate coordinate sets from existing box faces
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_data['boxes']:
            x_vals.add(px + pw)
            y_vals.add(py + ph)
            z_vals.add(pz + pd)
        
        # Filter to valid starting positions
        x_vals = sorted([x for x in x_vals if x + w <= W])
        y_vals = sorted([y for y in y_vals if y + h <= H])
        z_vals = sorted([z for z in z_vals if z + d <= D])
        
        if not x_vals or not y_vals or not z_vals:
            return None
        
        # Try positions in order of compactness (minimize x+y+z)
        best_pos = None
        best_cost = float('inf')
        
        grid = bin_data['grid']
        
        for z in z_vals:
            for y in y_vals:
                for x in x_vals:
                    if can_place_grid(grid, x, y, z, w, h, d):
                        cost = x + y + z
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                        if cost >= best_cost:
                            # Early exit: if we're iterating in increasing cost order,
                            # we can't do better once cost exceeds best
                            pass
        
        return best_pos
    
    def compute_score(bins_list, placement_dict):
        """
        Compute the score: B + 0.1*(1 - last-bin occupied volume fraction).
        B is the number of occupied bins.
        """
        if not bins_list:
            return 0.0
        num_bins = len(bins_list)
        # Find the last bin (largest label)
        last_bin_idx = max(x[0] for x in placement_dict.values()) if placement_dict else 0
        last_bin_volume = bins_list[last_bin_idx]['volume'] if last_bin_idx < len(bins_list) else 0
        last_bin_frac = last_bin_volume / container_vol if container_vol > 0 else 0
        return num_bins + 0.1 * (1 - last_bin_frac)
    
    # Process boxes in decreasing volume order
    for idx, vol, w, h, d in indexed_boxes:
        # Time check
        if time.time() > deadline:
            # Fallback: place remaining boxes with simple placement
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                if idx2 in placement:
                    continue
                placed = False
                # Try existing bins
                for b_idx in range(len(bins)):
                    spot = find_best_spot(bins[b_idx], w2, h2, d2)
                    if spot is not None:
                        x, y, z = spot
                        place_in_bin(bins[b_idx], x, y, z, w2, h2, d2)
                        placement[idx2] = (b_idx, x, y, z)
                        placed = True
                        break
                if not placed:
                    # Create new bin
                    bins.append(create_bin())
                    place_in_bin(bins[-1], 0, 0, 0, w2, h2, d2)
                    placement[idx2] = (len(bins) - 1, 0, 0, 0)
            
            # Build result
            result = []
            for i in range(n):
                b_idx, x, y, z = placement[i]
                w_i, h_i, d_i = boxes[i]
                result.append((b_idx, x, y, z, w_i, h_i, d_i))
            return result
        
        # Try to place in existing bins using best-fit
        best_bin = -1
        best_x, best_y, best_z = 0, 0, 0
        best_cost = float('inf')
        
        for b_idx in range(len(bins)):
            spot = find_best_spot(bins[b_idx], w, h, d)
            if spot is not None:
                x, y, z = spot
                cost = x + y + z
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b_idx
                    best_x, best_y, best_z = x, y, z
        
        if best_bin == -1:
            # Create new bin
            bins.append(create_bin())
            best_bin = len(bins) - 1
            best_x, best_y, best_z = 0, 0, 0
        
        # Place the box
        place_in_bin(bins[best_bin], best_x, best_y, best_z, w, h, d)
        placement[idx] = (best_bin, best_x, best_y, best_z)
    
    # Local improvement: try to reduce number of bins or improve last-bin fill
    max_improvement_passes = 5
    for pass_num in range(max_improvement_passes):
        if time.time() > deadline - 20.0:
            break
        
        improved = False
        # Try to move boxes from the last bin to other bins
        last_bin_idx = max(x[0] for x in placement.values()) if placement else -1
        if last_bin_idx < 0 or last_bin_idx >= len(bins):
            break
        
        # Get boxes in the last bin
        last_bin_boxes = list(bins[last_bin_idx]['boxes'])
        
        for (bx, by, bz, bw, bh, bd) in last_bin_boxes:
            idx = None
            # Find the original index for this box
            for i, (b_idx, x, y, z) in placement.items():
                if b_idx == last_bin_idx and x == bx and y == by and z == bz:
                    idx = i
                    break
            if idx is None:
                continue
            
            # Try to move this box to another bin
            for target_bin in range(len(bins)):
                if target_bin == last_bin_idx:
                    continue
                
                # Temporarily remove box from last bin
                unplace_in_bin(bins[last_bin_idx], bx, by, bz, bw, bh, bd)
                
                # Try to place in target bin
                spot = find_best_spot(bins[target_bin], bw, bh, bd)
                if spot is not None:
                    old_score = compute_score(bins, placement)
                    
                    # Perform the move
                    tx, ty, tz = spot
                    place_in_bin(bins[target_bin], tx, ty, tz, bw, bh, bd)
                    placement[idx] = (target_bin, tx, ty, tz)
                    
                    new_score = compute_score(bins, placement)
                    
                    if new_score < old_score - 1e-9:
                        improved = True
                        break
                    else:
                        # Revert the move
                        unplace_in_bin(bins[target_bin], tx, ty, tz, bw, bh, bd)
                        place_in_bin(bins[last_bin_idx], bx, by, bz, bw, bh, bd)
                        placement[idx] = (last_bin_idx, bx, by, bz)
                
                if not spot:
                    # Restore the box in last bin if it wasn't placed
                    place_in_bin(bins[last_bin_idx], bx, by, bz, bw, bh, bd)
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
# EVOLVE-BLOCK-END
