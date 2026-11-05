"""3D Bin Packing using Maximal Space Partitioning (Best-Fit Decreasing)."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending for better packing (Best-Fit Decreasing)
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Each bin is a list of free spaces: (x, y, z, w, h, d)
    bins = []
    # placements[original_index] = (bin_id, x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    def make_initial_bin():
        return [(0, 0, 0, W, H, D)]
    
    def split_space(space, box):
        """
        Given a free space and a box placed at (bx, by, bz) with dims (bw, bh, bd),
        split the space into up to 6 non-overlapping sub-spaces.
        Returns list of new free spaces.
        """
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz = box[1], box[2], box[3]
        bw, bh, bd = box[4], box[5], box[6]
        
        # Check if box is fully inside space
        if bx < sx or bx + bw > sx + sw or by < sy or by + bh > sy + sh or bz < sz or bz + bd > sz + sd:
            # This shouldn't happen if we only place in valid spaces, but just in case
            return [space]
        
        new_spaces = []
        
        # Split along X
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - (bx + bw), sh, sd))
        
        # Split along Y (within the x-range of the box)
        x_start = max(sx, bx)
        x_end = min(sx + sw, bx + bw)
        if by > sy:
            new_spaces.append((x_start, sy, sz, x_end - x_start, by - sy, sd))
        if by + bh < sy + sh:
            new_spaces.append((x_start, by + bh, sz, x_end - x_start, sy + sh - (by + bh), sd))
        
        # Split along Z (within the x-y range of the box)
        y_start = max(sy, by)
        y_end = min(sy + sh, by + bh)
        if bz > sz:
            new_spaces.append((x_start, y_start, sz, x_end - x_start, y_end - y_start, bz - sz))
        if bz + bd < sz + sd:
            new_spaces.append((x_start, y_start, bz + bd, x_end - x_start, y_end - y_start, sz + sd - (bz + bd)))
        
        return new_spaces
    
    def find_best_fit(bin_spaces, w, h, d):
        """
        Find the best fitting space in the bin for a box of dimensions (w, h, d).
        Returns (space_index, x, y, z) or (None, None, None, None) if no fit.
        Best fit: minimize wasted volume in the smallest dimension that allows placement.
        We try to minimize the "slack" which is the sum of remaining space in each dimension.
        """
        best_idx = -1
        best_slack = float('inf')
        best_pos = (0, 0, 0)
        
        for i, (sx, sy, sz, sw, sh, sd) in enumerate(bin_spaces):
            if w <= sw and h <= sh and d <= sd:
                # Calculate slack: prefer tightest fit
                # We can place anywhere in the space; let's place at origin of space for simplicity
                # but better to minimize max slack or sum of slacks
                slack_x = sw - w
                slack_y = sh - h
                slack_z = sd - d
                # Use a weighted sum or max to prefer tight fits
                # Let's use the product of slacks or sum. Sum is simpler.
                # Actually, for best-fit, we want the space where the box fits with least waste.
                # A common metric is the minimum of the three slacks, or the sum.
                # Let's use the sum of slacks as a proxy for waste.
                total_slack = slack_x + slack_y + slack_z
                
                if total_slack < best_slack:
                    best_slack = total_slack
                    best_idx = i
                    best_pos = (sx, sy, sz)
        
        if best_idx == -1:
            return None, None, None, None
        return best_idx, best_pos[0], best_pos[1], best_pos[2]
    
    # Main loop
    for orig_idx, (w, h, d) in indexed_boxes:
        placed = False
        
        # Try existing bins first (Best Fit)
        best_bin = -1
        best_space_idx = -1
        best_x, best_y, best_z = 0, 0, 0
        best_global_slack = float('inf')
        
        for b in range(len(bins)):
            space_idx, px, py, pz = find_best_fit(bins[b], w, h, d)
            if space_idx is not None:
                # Calculate slack for comparison across bins
                space = bins[b][space_idx]
                slack = (space[3] - w) + (space[4] - h) + (space[5] - d)
                if slack < best_global_slack:
                    best_global_slack = slack
                    best_bin = b
                    best_space_idx = space_idx
                    best_x, best_y, best_z = px, py, pz
        
        if best_bin != -1:
            # Place in existing bin
            bin_spaces = bins[best_bin]
            space = bin_spaces[best_space_idx]
            
            # Create the box tuple for splitting: (dummy, x, y, z, w, h, d)
            box_tuple = (0, best_x, best_y, best_z, w, h, d)
            
            # Split the space
            new_spaces = split_space(space, box_tuple)
            
            # Replace the old space with the new ones
            bin_spaces[best_space_idx:best_space_idx+1] = new_spaces
            
            placements[orig_idx] = (best_bin, best_x, best_y, best_z, w, h, d)
            placed = True
        
        if not placed:
            # Open a new bin
            new_bin = make_initial_bin()
            bin_id = len(bins)
            bins.append(new_bin)
            
            # Place at origin
            x, y, z = 0, 0, 0
            box_tuple = (0, x, y, z, w, h, d)
            new_spaces = split_space(new_bin[0], box_tuple)
            new_bin[0:1] = new_spaces
            
            placements[orig_idx] = (bin_id, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
