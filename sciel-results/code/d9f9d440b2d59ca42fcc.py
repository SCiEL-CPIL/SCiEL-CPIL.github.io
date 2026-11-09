"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _pack_shelf(boxes, container):
    """Shelf heuristic: rows along x, shelves along y, layers along z."""
    W, H, D = container
    placements = []
    b = 0
    x = y = z = 0
    row_h = 0
    layer_d = 0
    for w, h, d in boxes:
        # Try to place in current position
        if x + w > W:
            x = 0
            y += row_h
            row_h = 0
        if y + h > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + d > D:
            b += 1
            x = y = z = 0
            row_h = 0
            layer_d = 0
        placements.append((b, x, y, z, w, h, d))
        x += w
        row_h = max(row_h, h)
        layer_d = max(layer_d, d)
    return placements

def _pack_best_fit(boxes, container, order='first'):
    """Best-fit decreasing by volume, using a simple grid-free placement search."""
    W, H, D = container
    n = len(boxes)
    
    if order == 'first':
        indices = list(range(n))
    else:
        # Sort by volume descending
        vol = [w * h * d for w, h, d in boxes]
        indices = sorted(range(n), key=lambda i: -vol[i])
    
    # For each bin, maintain a list of occupied boxes
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def fits_in_bin(bin_boxes, x, y, z, w, h, d):
        """Check if box at (x,y,z) with dims (w,h,d) fits in W,H,D and doesn't overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bin_boxes:
            # Check overlap
            if (x < ox + ow and ox < x + w and
                y < oy + oh and oy < y + h and
                z < oz + od and oz < z + d):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find a valid position in the bin for a box with dims (w,h,d)."""
        # Try a set of candidate positions: bottom-left-front corner and offsets
        # Simple approach: try x from 0 to W-w step 1, y from 0 to H-h, z from 0 to D-d
        # This is too slow. Use a smarter approach.
        
        # Collect candidate x, y, z values
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (ox, oy, oz, ow, oh, od) in bin_boxes:
            x_candidates.add(ox + ow)
            y_candidates.add(oy + oh)
            z_candidates.add(oz + od)
        
        # Filter candidates within bounds
        x_candidates = [xc for xc in x_candidates if xc + w <= W]
        y_candidates = [yc for yc in y_candidates if yc + h <= H]
        z_candidates = [zc for zc in z_candidates if zc + d <= D]
        
        # Sort by priority (lower is better)
        x_candidates.sort()
        y_candidates.sort()
        z_candidates.sort()
        
        for xc in x_candidates:
            for yc in y_candidates:
                for zc in z_candidates:
                    if fits_in_bin(bin_boxes, xc, yc, zc, w, h, d):
                        return (xc, yc, zc)
        
        return None
    
    results = [None] * n  # (bin_idx, x, y, z) for each original box index
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((x, y, z, w, h, d))
                results[idx] = (b_idx, x, y, z)
                placed = True
                break
        
        if not placed:
            # Start a new bin
            b_idx = len(bins)
            bins.append([])
            # Place at origin
            bins[b_idx].append((0, 0, 0, w, h, d))
            results[idx] = (b_idx, 0, 0, 0)
    
    # Build placements
    placements = []
    for i in range(n):
        b_idx, x, y, z = results[i]
        w, h, d = boxes[i]
        placements.append((b_idx, x, y, z, w, h, d))
    
    return placements

def _pack_bfd(boxes, container):
    """Best Fit Decreasing: sort by volume descending, place each in the bin where it fits with least remaining space."""
    W, H, D = container
    n = len(boxes)
    vol = [w * h * d for w, h, d in boxes]
    indices = sorted(range(n), key=lambda i: -vol[i])
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def fits_in_bin(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bin_boxes:
            if (x < ox + ow and ox < x + w and
                y < oy + oh and oy < y + h and
                z < oz + od and oz < z + d):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (ox, oy, oz, ow, oh, od) in bin_boxes:
            x_candidates.add(ox + ow)
            y_candidates.add(oy + oh)
            z_candidates.add(oz + od)
        
        x_candidates = [xc for xc in x_candidates if xc + w <= W]
        y_candidates = [yc for yc in y_candidates if yc + h <= H]
        z_candidates = [zc for zc in z_candidates if zc + d <= D]
        
        x_candidates.sort()
        y_candidates.sort()
        z_candidates.sort()
        
        for xc in x_candidates:
            for yc in y_candidates:
                for zc in z_candidates:
                    if fits_in_bin(bin_boxes, xc, yc, zc, w, h, d):
                        return (xc, yc, zc)
        return None
    
    results = [None] * n
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((x, y, z, w, h, d))
                results[idx] = (b_idx, x, y, z)
                placed = True
                break
        
        if not placed:
            b_idx = len(bins)
            bins.append([])
            bins[b_idx].append((0, 0, 0, w, h, d))
            results[idx] = (b_idx, 0, 0, 0)
    
    placements = []
    for i in range(n):
        b_idx, x, y, z = results[i]
        w, h, d = boxes[i]
        placements.append((b_idx, x, y, z, w, h, d))
    
    return placements

def _compute_score(placements, boxes, container):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return 0.0
    
    W, H, D = container
    container_vol = W * H * D
    
    # Find occupied bins
    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        if b not in bin_volumes:
            bin_volumes[b] = 0
        bin_volumes[b] += w * h * d
    
    B = len(bin_volumes)
    
    if B == 0:
        return 0.0
    
    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0
    
    return B + 0.1 * (1.0 - last_frac)

def _pack_with_time_budget(boxes, container, deadline):
    """Try multiple heuristics within the time budget, return best placement."""
    start_time = time.time()
    
    best_placements = None
    best_score = float('inf')
    
    # Strategy 1: Shelf (fast)
    placements = _pack_shelf(boxes, container)
    score = _compute_score(placements, boxes, container)
    if score < best_score:
        best_score = score
        best_placements = placements
    
    if time.time() > deadline - 5:
        return best_placements
    
    # Strategy 2: BFD (slower but often better)
    placements = _pack_bfd(boxes, container)
    score = _compute_score(placements, boxes, container)
    if score < best_score:
        best_score = score
        best_placements = placements
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 3: Try random restarts of BFD with different orderings
    n = len(boxes)
    vol = [w * h * d for w, h, d in boxes]
    
    for attempt in range(5):
        if time.time() > deadline - 1:
            break
        
        # Random perturbation of the ordering
        if attempt == 0:
            indices = sorted(range(n), key=lambda i: -vol[i])
        else:
            indices = list(range(n))
            random.shuffle(indices)
            # Sort by a mix of volume and random
            indices.sort(key=lambda i: (-vol[i], random.random()))
        
        # Use BFD logic with this ordering
        W, H, D = container
        bins = []
        
        def fits_in_bin(bin_boxes, x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            for (ox, oy, oz, ow, oh, od) in bin_boxes:
                if (x < ox + ow and ox < x + w and
                    y < oy + oh and oy < y + h and
                    z < oz + od and oz < z + d):
                    return False
            return True
        
        def find_position(bin_boxes, w, h, d):
            x_candidates = set([0])
            y_candidates = set([0])
            z_candidates = set([0])
            
            for (ox, oy, oz, ow, oh, od) in bin_boxes:
                x_candidates.add(ox + ow)
                y_candidates.add(oy + oh)
                z_candidates.add(oz + od)
            
            x_candidates = [xc for xc in x_candidates if xc + w <= W]
            y_candidates = [yc for yc in y_candidates if yc + h <= H]
            z_candidates = [zc for zc in z_candidates if zc + d <= D]
            
            x_candidates.sort()
            y_candidates.sort()
            z_candidates.sort()
            
            for xc in x_candidates:
                for yc in y_candidates:
                    for zc in z_candidates:
                        if fits_in_bin(bin_boxes, xc, yc, zc, w, h, d):
                            return (xc, yc, zc)
            return None
        
        results = [None] * n
        
        for idx in indices:
            w, h, d = boxes[idx]
            placed = False
            
            for b_idx, bin_boxes in enumerate(bins):
                pos = find_position(bin_boxes, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bin_boxes.append((x, y, z, w, h, d))
                    results[idx] = (b_idx, x, y, z)
                    placed = True
                    break
            
            if not placed:
                b_idx = len(bins)
                bins.append([])
                bins[b_idx].append((0, 0, 0, w, h, d))
                results[idx] = (b_idx, 0, 0, 0)
        
        placements = []
        for i in range(n):
            b_idx, x, y, z = results[i]
            w, h, d = boxes[i]
            placements.append((b_idx, x, y, z, w, h, d))
        
        score = _compute_score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    return best_placements

def pack(boxes, container):
    """
    Place every box into numbered bins.
    Returns one placement (bin, x, y, z, w, h, d) per box.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    if not boxes:
        return []
    
    # Use the time-budgeted approach
    placements = _pack_with_time_budget(boxes, container, deadline)
    
    return placements

# EVOLVE-BLOCK-END
