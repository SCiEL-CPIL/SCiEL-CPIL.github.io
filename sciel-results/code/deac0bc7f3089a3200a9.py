"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Place boxes (w, h, d) into bins of size (W, H, D) without rotation or overlap.
    Returns list of (bin, x, y, z, w, h, d) preserving box order.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    W, H, D = container
    if len(boxes) == 0:
        return []

    # Precompute box volumes and sort indices by descending volume for placement order
    # But we must return in original order, so we track original index
    box_data = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    # Sort by volume descending, then by max dimension descending
    sorted_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], max(boxes[i])), reverse=True)

    # We'll use a best-fit decreasing approach with a spatial data structure per bin
    # For each bin, maintain a list of placed boxes and use a simple grid or interval tree
    # Given the small size (40 boxes), we can use a simple O(n) overlap check per placement

    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap: two boxes overlap if they intersect in all three dimensions
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        return True

    def find_position(bin_boxes, w, h, d):
        """
        Find a position for box (w,h,d) in a bin using a shelf-based heuristic.
        Returns (x, y, z) or None if no position found.
        """
        # Generate candidate positions using shelf heuristic
        # Shelves are horizontal planes. We try to place on existing shelves or new shelves.
        
        # Collect all y-levels where a shelf could start: 0 and top of each placed box
        y_candidates = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if py + ph <= H:
                y_candidates.append(py + ph)
        # Also consider z-levels for layers
        z_candidates = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if pz + pd <= D:
                z_candidates.append(pz + pd)
        
        # Sort candidates
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for z in z_candidates:
            if z + d > D:
                continue
            for y in y_candidates:
                if y + h > H:
                    continue
                # Find available x positions on this shelf (y, z)
                # Collect x-intervals occupied on this shelf
                occupied = []
                for (px, py, pz, pw, ph, pd) in bin_boxes:
                    # Check if this box is on the same shelf (y and z overlap with our candidate)
                    if not (y + h <= py or py + ph <= y):
                        if not (z + d <= pz or pz + pd <= z):
                            # This box occupies some x-space on our shelf
                            occupied.append((px, px + pw))
                
                # Sort occupied intervals
                occupied.sort()
                
                # Find gaps
                gaps = []
                prev_end = 0
                for (s, e) in occupied:
                    if s > prev_end:
                        gaps.append((prev_end, s))
                    prev_end = max(prev_end, e)
                if prev_end < W:
                    gaps.append((prev_end, W))
                
                # Check if w fits in any gap
                for (gs, ge) in gaps:
                    if ge - gs >= w:
                        # Place at gs (leftmost available)
                        pos = (gs, y, z)
                        # Score: prefer lower y, then lower z, then lower x (compact)
                        score = y * 10000 + z * 100 + gs
                        if score < best_score:
                            best_score = score
                            best_pos = pos
        
        return best_pos

    # Main packing loop
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = [None] * len(boxes)  # (bin_idx, x, y, z) for each original box index
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each
            for j in range(idx, len(boxes)):
                if placement[j] is None:
                    bins.append([])
                    bins[-1].append((0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2]))
                    placement[j] = (len(bins) - 1, 0, 0, 0)
            break
        
        # Try existing bins first (best-fit: prefer bin with least remaining space)
        for b_idx in range(len(bins)):
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                placement[idx] = (b_idx, x, y, z)
                placed = True
                break
        
        if not placed:
            # Open a new bin
            bins.append([(0, 0, 0, w, h, d)])
            placement[idx] = (len(bins) - 1, 0, 0, 0)
    
    # Build result in original order
    result = []
    for i in range(len(boxes)):
        if placement[i] is None:
            # Should not happen, but safety
            bins.append([])
            bins[-1].append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            placement[i] = (len(bins) - 1, 0, 0, 0)
        b, x, y, z = placement[i]
        w, h, d = boxes[i]
        result.append((b, x, y, z, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
