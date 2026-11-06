"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    3D bin packing using a best-fit decreasing strategy with a simple 3D
    placement search that tries to minimize the number of bins used and
    then minimizes the empty space in the last bin.
    
    Returns a list of (bin_index, x, y, z, w, h, d) for each box in the
    original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending for placement, but remember original indices
    # We'll process in sorted order but assign to original positions
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])  # sort by volume descending
    
    # We'll maintain a list of bins. Each bin has a list of placed boxes (x, y, z, w, h, d)
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def find_placement(bin_boxes, w, h, d):
        """
        Try to find a position (x, y, z) in the given bin for a box of size (w, h, d).
        Uses a simple grid-based approach: try to place at the lowest possible z,
        then lowest y, then lowest x that doesn't overlap.
        
        Returns (x, y, z) or None if no placement found.
        """
        # Generate candidate positions: for each existing box, consider positions
        # just to the right, above, and in front of it, plus the origin.
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Right
            candidates.append((px + pw, py, pz))
            # Above
            candidates.append((px, py + ph, pz))
            # In front
            candidates.append((px, py, pz + pd))
            # Also try combinations: right and above, etc.
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        
        # Remove duplicates and filter valid positions
        seen = set()
        valid_candidates = []
        for (x, y, z) in candidates:
            if (x, y, z) in seen:
                continue
            seen.add((x, y, z))
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check for overlap with all existing boxes
            overlap = False
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    overlap = True
                    break
            if not overlap:
                valid_candidates.append((x, y, z))
        
        if not valid_candidates:
            return None
        
        # Choose the candidate that minimizes z, then y, then x (lowest position)
        best = min(valid_candidates, key=lambda pos: (pos[2], pos[1], pos[0]))
        return best
    
    # Process boxes in sorted order
    original_placements = [None] * n
    for i, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
        placed = False
        # Try to place in existing bins first (best fit: minimize wasted space)
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for bin_idx in range(len(bins)):
            pos = find_placement(bins[bin_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: prefer lower z, then lower y, then lower x
                score = (z, y, x)
                if score < best_score:
                    best_score = score
                    best_bin = bin_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            original_placements[orig_idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Start a new bin
            bin_idx = len(bins)
            pos = find_placement([], w, h, d)
            if pos is None:
                # Box doesn't fit in a new bin? This shouldn't happen if box fits in container
                # But handle gracefully
                pos = (0, 0, 0)
            x, y, z = pos
            bins.append([(x, y, z, w, h, d)])
            original_placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
    
    return original_placements

# EVOLVE-BLOCK-END
