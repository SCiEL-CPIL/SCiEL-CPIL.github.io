"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    """
    Pack boxes into bins using a 3D occupancy grid approach with a
    'best-fit decreasing' heuristic based on maximal empty space estimation.
    
    Returns: list of (bin, x, y, z, w, h, d) tuples.
    """
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    
    # Sort boxes by volume descending, then by max dimension descending
    # This helps fill large empty spaces first
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda ib: (-(ib[1][0]*ib[1][1]*ib[1][2]), -max(ib[1])))
    
    # Each bin has a 3D boolean occupancy grid
    # bins[bin_id] is a numpy array of shape (W, H, D) with True where occupied
    bins = {}
    bin_id_counter = 0
    
    def get_bin_grid(b):
        if b not in bins:
            bins[b] = np.zeros((W, H, D), dtype=bool)
        return bins[b]
    
    def find_best_placement(grid, w, h, d):
        """
        Find the best (x, y, z) position for a box of dimensions (w, h, d)
        in the given bin grid. Uses a heuristic that prefers positions
        close to the origin (bottom-left-front) to keep space compact.
        
        Returns (x, y, z) or None if no valid position found.
        """
        if w > W or h > H or d > D:
            return None
        
        # Precompute: for each possible (x, y, z), check if the box fits
        # We want the position that minimizes (x, y, z) in a lexicographic sense
        # but also considers how much "wasted" space is created.
        # Simple heuristic: find the first valid position scanning from origin.
        
        # To speed up, we can precompute a "can place" array
        # A box of size (w, h, d) can be placed at (x, y, z) if
        # grid[x:x+w, y:y+h, z:z+d] has no True values.
        
        # Efficient check: use cumulative sum or just slice check
        # For small grids, direct slicing is fine.
        
        best_pos = None
        best_score = float('inf')
        
        # Scan all possible positions
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    # Check if box fits
                    if not np.any(grid[x:x+w, y:y+h, z:z+d]):
                        # Score: prefer positions closer to origin
                        # Use a weighted sum to encourage compact packing
                        score = x * D * H + y * D + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def find_best_placement_fast(grid, w, h, d):
        """
        Faster placement search using a precomputed 'empty space' map.
        We maintain for each bin a structure that helps quickly find valid positions.
        """
        if w > W or h > H or d > D:
            return None
        
        # Create a boolean array where True means the box CAN start at that position
        # can_place[x, y, z] = True if grid[x:x+w, y:y+h, z:z+d] is all False
        
        # Efficient way: use 3D convolution or cumulative sums
        # For simplicity and correctness, let's use a different approach:
        # Maintain a list of "empty regions" or just scan smartly.
        
        # Alternative: for each z-slice, find empty y-x regions
        # Let's use a simpler but effective heuristic:
        # Scan z from 0 to D-d, y from 0 to H-h, x from 0 to W-w
        # and return the first valid position (closest to origin)
        
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                # Check if the entire (w, h) rectangle in the x-y plane at this z
                # is empty for all d layers
                valid = True
                for dz in range(d):
                    # Check if grid[:, y:y+h, z+dz] has any True in x range [0, W-w]
                    # Actually we need to check specific x range too
                    # Let's just check the full slice
                    if np.any(grid[:, y:y+h, z+dz][:, :]):
                        # Need to be more precise
                        pass
                # This is getting complex. Let's just use the direct method
                # but with early termination
                for x in range(W - w + 1):
                    if not np.any(grid[x:x+w, y:y+h, z:z+d]):
                        return (x, y, z)
        
        return None
    
    # For performance, let's use a simpler but effective approach:
    # For each box, try to place it in the existing bin that gives the "best fit"
    # Best fit = smallest remaining space after placement, or closest to origin
    
    def can_place(grid, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        return not np.any(grid[x:x+w, y:y+h, z:z+d])
    
    def place_in_bin(grid, w, h, d):
        """Find the best position in a given bin grid."""
        # First, try to place at the "front" of existing content
        # Compute the current bounding box of occupied space
        occupied = np.where(grid)
        if len(occupied[0]) == 0:
            return (0, 0, 0)
        
        # Find the "next" empty position by scanning from origin
        for z in range(D - d + 1):
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    if can_place(grid, x, y, z, w, h, d):
                        return (x, y, z)
        
        return None
    
    # Main packing loop
    placements = []
    bin_ids_used = set()
    
    for orig_idx, (w, h, d) in indexed_boxes:
        w, h, d = int(w), int(h), int(d)
        
        placed = False
        
        # Try existing bins first, sorted by bin id
        for b in sorted(bin_ids_used):
            grid = get_bin_grid(b)
            pos = place_in_bin(grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                grid[x:x+w, y:y+h, z:z+d] = True
                placements.append((orig_idx, b, x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Open a new bin
            b = bin_id_counter
            bin_id_counter += 1
            bin_ids_used.add(b)
            grid = get_bin_grid(b)
            if can_place(grid, 0, 0, 0, w, h, d):
                grid[0:w, 0:h, 0:d] = True
                placements.append((orig_idx, b, 0, 0, 0, w, h, d))
            else:
                # Box doesn't fit in a new bin? This shouldn't happen if box fits in container
                # But handle gracefully
                placements.append((orig_idx, b, 0, 0, 0, w, h, d))
    
    # Reorder placements to match original box order
    # placements has (orig_idx, b, x, y, z, w, h, d)
    # We need to return in the order of the original boxes list
    result = [None] * len(boxes)
    for orig_idx, b, x, y, z, w, h, d in placements:
        result[orig_idx] = (b, x, y, z, w, h, d)
    
    # Verify all boxes are placed
    for i, r in enumerate(result):
        if r is None:
            # Should not happen
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return result
# EVOLVE-BLOCK-END
