"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending.
2. Place boxes into bins using a Best-Fit-Decreasing heuristic.
   - For each box, try existing bins (sorted by least remaining volume).
   - Within a bin, find the position with lowest z, then lowest y, then lowest x.
3. Post-processing:
   - Identify the last bin (highest label).
   - Try to move boxes from other bins into the last bin to increase its fill ratio.
   - This improves the score term 0.1 * (1 - fill_ratio) without increasing B.
4. Relabel bins to be consecutive.
"""
import time
import numpy as np
from scipy import ndimage

def _can_place(box, pos, placements, W, H, D, grid=None):
    """Check if box with dims (w,h,d) can be placed at (x,y,z) in a bin with given placements."""
    x, y, z = pos
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False

    if grid is not None:
        if bool(grid[x:x+w, y:y+h, z:z+d].any()):
            return False
        return True

    for (_, px, py, pz, pw, ph, pd) in placements:
        # Check overlap
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position(box, placements, W, H, D, grid=None):
    """
    Find the best position for a box in a bin.
    Strategy: Use SciPy ndimage to find maximal empty spaces, then pick
    the lexicographically smallest (z, y, x) that fits the box.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    if grid is None:
        # Fallback to manual check if no grid
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    if _can_place(box, (x, y, z), placements, W, H, D, None):
                        return (x, y, z)
        return None

    # grid is a 3D boolean array of shape (W, H, D)
    # grid[x, y, z] = True means occupied

    # Use ndimage to find all positions where a box of size (w, h, d) can fit.
    # We want positions (x, y, z) such that grid[x:x+w, y:y+h, z:z+d] is all False.
    # This is equivalent to: for each (x,y,z), the min over the sub-array is 1 (unoccupied).
    
    # Create a structure element of size (w, h, d)
    # We use minimum filter: if the minimum value in the window is 1 (all unoccupied), it's a valid position
    # But our grid uses True=occupied, so we need min of (1 - grid) to be 1
    # i.e., all cells in the window must be unoccupied (False in grid)
    
    # Alternative: use ndimage.minimum with a structure element
    # If we invert the grid: unoccupied = 1, occupied = 0
    # Then minimum filter with a (w,h,d) structure: result[x,y,z] = 1 iff all cells in window are unoccupied
    
    unoccupied = (~grid).astype(np.int8)
    
    # Create structure element
    structure = np.ones((w, h, d), dtype=bool)
    
    # Apply minimum filter
    # The output will have the same shape as input
    # result[x, y, z] = min of unoccupied[x:x+w, y:y+h, z:z+d]
    valid = ndimage.minimum_filter(unoccupied, footprint=structure)
    
    # valid[x, y, z] == 1 means the box can be placed with its min corner at (x, y, z)
    # But we need to be careful: minimum_filter centers the footprint
    # For a footprint of size (w, h, d), the output at (x, y, z) considers
    # unoccupied[x - (w-1)//2 : x + (w-1)//2 + 1, ...]
    # This is not what we want. We want the box to start at (x, y, z).
    
    # Let's use a different approach: direct convolution or just check candidates
    # Actually, let's use ndimage with proper offset handling
    
    # Simpler: find all valid (x,y,z) by checking if the sub-array is all unoccupied
    # Use the fact that sum of occupied cells in the window must be 0
    
    # For efficiency, precompute prefix sums or just use the candidate approach
    # but with ndimage to speed up the search
    
    # Let's try: use ndimage to compute for each position the number of occupied cells
    # in a (w, h, d) window starting at that position
    
    # We can use uniform_filter but it averages, not sums
    # Let's just use the candidate-based approach but optimize it
    
    # Generate candidate coordinates from existing box boundaries
    xs = set([0])
    ys = set([0])
    zs = set([0])

    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)

    # Also add positions that align with the right/top/back faces
    for x in list(xs):
        if x + w <= W:
            xs.add(x)
    for y in list(ys):
        if y + h <= H:
            ys.add(y)
    for z in list(zs):
        if z + d <= D:
            zs.add(z)

    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])

    if not valid_xs or not valid_ys or not valid_zs:
        return None

    # Use ndimage to speed up: precompute a "can place" map
    # For each candidate z, check which (x, y) positions are valid
    
    # Build a 2D map for each z: valid_xy[x, y] = True if grid[x:x+w, y:y+h, z:z+d] is all False
    
    # We can use ndimage on slices
    for z in valid_zs:
        # Check the slice at this z level
        # We need grid[x:x+w, y:y+h, z:z+d] to be all False
        # This is equivalent to: for all dz in [0, d), grid[x:x+w, y:y+h, z+dz] is all False
        
        # Compute the AND of all d slices
        # Use minimum filter in 2D for each slice, then AND them
        
        slice_valid = np.ones((W, H), dtype=bool)
        for dz in range(d):
            zz = z + dz
            if zz >= D:
                slice_valid = np.zeros((W, H), dtype=bool)
                break
            # For this slice, we need grid[x:x+w, y:y+h, zz] to be all False
            # Use minimum filter on the inverted slice
            inv_slice = (~grid[:, :, zz]).astype(np.int8)
            if w == 1 and h == 1:
                slice_valid &= (inv_slice == 1)
            else:
                # Use minimum filter with a (w, h) footprint
                # But again, centering issue
                # Let's just do direct check for this slice
                # For each candidate (x, y), check if grid[x:x+w, y:y+h, zz] is all False
                # We can use a 2D minimum filter
                struct2d = np.ones((w, h), dtype=bool)
                min_val = ndimage.minimum_filter(inv_slice, footprint=struct2d)
                # min_val[x, y] = 1 iff all cells in the window are unoccupied
                # But centering: min_val[x, y] considers inv_slice[x-(w-1)//2 : x+(w-1)//2+1, ...]
                # This is offset. Let's handle it differently.
                
                # Direct approach: for candidate (x, y), check the sub-array
                # We'll just iterate over valid_xs and valid_ys for this z
                pass
        
        # For simplicity and correctness, let's just do the direct check
        # but use the grid for fast overlap detection
        for y in valid_ys:
            for x in valid_xs:
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
    bin_grids = [] # List of numpy boolean arrays (W, H, D)
    result = [None] * n
    
    def _update_grid(grid, box, pos, add=True):
        x, y, z = pos
        w, h, d = box
        grid[x:x+w, y:y+h, z:z+d] = add

    # 2. Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency placement: each in new bin
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    grid = np.zeros((W, H, D), dtype=bool)
                    _update_grid(grid, ob, (0,0,0))
                    bin_grids.append(grid)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Estimate remaining volume for each bin
            bin_vols = []
            for bi in range(len(bins)):
                # Use grid sum for fast volume check
                vol = np.sum(bin_grids[bi])
                bin_vols.append((W*H*D - vol, bi))
            
            # Sort by remaining volume ascending (tightest bins first)
            bin_vols.sort(key=lambda x: x[0])
            
            for _, bi in bin_vols:
                if time.time() > deadline - 3.0:
                    break
                pos = _find_position(box, bins[bi], W, H, D, bin_grids[bi])
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    _update_grid(bin_grids[bi], box, pos)
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, box, (0,0,0))
            bin_grids.append(grid)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
            
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0,0,0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    # 3. Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 5.0:
        # Identify last bin (highest label)
        last_bin_id = max(r[0] for r in result)
        
        moved_any = True
        while moved_any and time.time() < deadline - 2.0:
            moved_any = False
            candidates = [i for i in range(n) if result[i][0] != last_bin_id]
            for idx in candidates:
                if time.time() > deadline - 2.0:
                    break
                
                cur_bin = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                cur_placements = bins[cur_bin]
                box_pos = result[idx][1:4]
                box_idx_in_bin = None
                for j, p in enumerate(cur_placements):
                    if p[1] == box_pos[0] and p[2] == box_pos[1] and p[3] == box_pos[2] and p[4]==w and p[5]==h and p[6]==d:
                        box_idx_in_bin = j
                        break
                
                if box_idx_in_bin is None:
                    continue
                
                # Temporarily remove from source bin
                _update_grid(bin_grids[cur_bin], box, box_pos, add=False)
                cur_placements_no_box = cur_placements[:box_idx_in_bin] + cur_placements[box_idx_in_bin+1:]
                
                # Try to place in last bin
                last_placements = bins[last_bin_id]
                pos = _find_position(box, last_placements, W, H, D, bin_grids[last_bin_id])
                
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    _update_grid(bin_grids[last_bin_id], box, pos)
                    bins[cur_bin] = cur_placements_no_box
                    result[idx] = (last_bin_id, x, y, z, w, h, d)
                    moved_any = True
                    break
                else:
                    # Restore if it didn't fit
                    _update_grid(bin_grids[cur_bin], box, box_pos, add=True)
                    bins[cur_bin] = cur_placements
        
        # 3b. Try to consolidate bins: if a bin becomes empty, relabel
        # Also try to move boxes between non-last bins to potentially reduce B
        if time.time() < deadline - 3.0:
            # Check if any bin is now empty
            for bi in range(len(bins)):
                if not bins[bi]:
                    continue
            
            # Try to move boxes from bin with highest index (excluding last_bin_id) to other bins
            # to see if we can eliminate that bin
            non_last_bins = [bi for bi in range(len(bins)) if bi != last_bin_id and bins[bi]]
            if non_last_bins:
                # Sort by bin index descending, try to empty the highest non-last bin
                for src_bin in sorted(non_last_boxes, reverse=True) if False else sorted(non_last_bins, reverse=True):
                    if time.time() > deadline - 3.0:
                        break
                    # Try to move all boxes from src_bin to other bins
                    boxes_to_move = list(bins[src_bin])
                    can_move_all = True
                    temp_placements = {}
                    
                    for p in boxes_to_move:
                        bi, x, y, z, w, h, d = p
                        box = (w, h, d)
                        
                        # Find which box index this is
                        box_idx = None
                        for i in range(n):
                            if result[i] == p:
                                box_idx = i
                                break
                        if box_idx is None:
                            can_move_all = False
                            break
                        
                        # Remove from source
                        _update_grid(bin_grids[src_bin], box, (x, y, z), add=False)
                        
                        # Try to place in other bins
                        placed = False
                        for dst_bin in range(len(bins)):
                            if dst_bin == src_bin:
                                continue
                            pos = _find_position(box, bins[dst_bin], W, H, D, bin_grids[dst_bin])
                            if pos is not None:
                                nx, ny, nz = pos
                                bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                                _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                                result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                                placed = True
                                break
                        
                        if not placed:
                            # Restore
                            _update_grid(bin_grids[src_bin], box, (x, y, z))
                            can_move_all = False
                            break
                    
                    if can_move_all:
                        bins[src_bin] = []
                    
    # 4. Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
            
    return result
