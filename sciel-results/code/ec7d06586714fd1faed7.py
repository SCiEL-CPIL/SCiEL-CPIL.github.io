"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Bin structure:
    # Each bin is a list of shelves.
    # Each shelf is a dict: { 'z': float, 'height': float, 'boxes': list of (x, y, w, h, orig_idx) }
    # Actually, we need to map back to original indices.
    # Let's store placements directly.
    
    bins = [] # List of lists of shelves
    # Each shelf: {'z': z_start, 'height': max_h, 'items': [(x, y, w, h, orig_idx), ...]}
    
    def get_2d_position(shelf_items, w, h):
        """
        Find a valid (x, y) position for a box of size (w, h) on a 2D plane of size (W, H),
        given a list of already placed items (x, y, w, h).
        Returns (x, y) or None.
        """
        candidates = [(0, 0)]
        for (ix, iy, iw, ih, _) in shelf_items:
            # Right of existing box
            if ix + iw + w <= W:
                candidates.append((ix + iw, iy))
            # Above existing box (in 2D sense, i.e., higher y)
            if iy + ih + h <= H:
                candidates.append((ix, iy + ih))
        
        # Also consider positions aligned with existing boxes' edges?
        # The simple "right" and "above" is often sufficient for First Fit.
        # To be more robust, we can also try (0, iy+ih) and (ix+iw, 0).
        for (ix, iy, iw, ih, _) in shelf_items:
            if iy + ih + h <= H:
                candidates.append((0, iy + ih))
            if ix + iw + w <= W:
                candidates.append((ix + iw, 0))
        
        # Sort candidates to prefer lower y, then lower x (compact front-left)
        # Or lower x then lower y?
        # Let's try (y, x) to fill the "front" (low y) first.
        candidates.sort(key=lambda p: (p[1], p[0]))
        
        for (x, y) in candidates:
            if x < 0 or y < 0:
                continue
            if x + w > W or y + h > H:
                continue
            
            # Check overlap
            overlap = False
            for (ix, iy, iw, ih, _) in shelf_items:
                if (x < ix + iw and x + w > ix and
                    y < iy + ih and y + h > iy):
                    overlap = True
                    break
            if not overlap:
                return (x, y)
        
        return None

    def try_place_in_bin(bin_shelves, w, h, d, orig_idx):
        """
        Try to place a box in an existing bin.
        Returns (shelf_index, x, y, z) if successful, else None.
        """
        # Try existing shelves
        for s_idx, shelf in enumerate(bin_shelves):
            z_start = shelf['z']
            # The box must fit in the vertical space of this shelf?
            # No, the shelf's height is determined by the max height of items on it.
            # If we place a box here, it occupies [z_start, z_start + h].
            # This is valid as long as it doesn't overlap in X/Y with other items on this shelf.
            # It doesn't matter if h > shelf['height'], the shelf height will just increase.
            
            pos = get_2d_position(shelf['items'], w, h)
            if pos is not None:
                x, y = pos
                # Update shelf height if needed
                if h > shelf['height']:
                    shelf['height'] = h
                shelf['items'].append((x, y, w, h, orig_idx))
                return (s_idx, x, y, z_start)
        
        # Try to create a new shelf at the top
        # Calculate current top z
        if not bin_shelves:
            top_z = 0
        else:
            # Shelves are stacked. The top z is the sum of heights?
            # No, we store 'z' explicitly.
            # The next shelf starts at the max (z + height) of existing shelves?
            # Since we stack them, they should be contiguous.
            # Let's just find the max z_end.
            top_z = max(s['z'] + s['height'] for s in bin_shelves)
            
        if top_z + d <= D:
            # Create new shelf
            new_shelf = {
                'z': top_z,
                'height': d,
                'items': [(0, 0, w, h, orig_idx)]
            }
            bin_shelves.append(new_shelf)
            return (len(bin_shelves) - 1, 0, 0, top_z)
        
        return None

    # Main loop
    placements = [None] * n
    
    for orig_idx, w, h, d in indexed_boxes:
        placed = False
        # Try existing bins
        for b_idx, bin_shelves in enumerate(bins):
            res = try_place_in_bin(bin_shelves, w, h, d, orig_idx)
            if res is not None:
                s_idx, x, y, z = res
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Create new bin
            new_bin = []
            new_bin.append({
                'z': 0,
                'height': d,
                'items': [(0, 0, w, h, orig_idx)]
            })
            bins.append(new_bin)
            placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
    return placements

# EVOLVE-BLOCK-END
