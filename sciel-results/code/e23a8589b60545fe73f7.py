"""Rewritten program for 3D Container Loading with stronger packing heuristics."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _all_orientations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for o in [(w,h,d),(w,d,h),(h,w,d),(h,d,w),(d,w,h),(d,h,w)]:
        if o not in seen:
            seen.add(o)
            result.append(o)
    return result

def _fits_at(x, y, z, w, h, d, W, H, D, placed):
    """Check if box (x,y,z,w,h,d) fits without overlapping any placed box."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return False
    return True

def _find_best_position(w, h, d, W, H, D, placed, max_z=None, max_y=None, max_x=None):
    """Find the 'lowest' position (min z, then min y, then min x) for a box.
    Uses candidate positions derived from existing box boundaries for efficiency.
    """
    # Build candidate x, y, z positions
    # x candidates: 0, and px+pw for each placed box
    x_candidates = [0]
    y_candidates = [0]
    z_candidates = [0]
    
    for (px, py, pz, pw, ph, pd) in placed:
        x_candidates.append(px + pw)
        y_candidates.append(py + ph)
        z_candidates.append(pz + pd)
    
    # Also add positions near the box dimensions
    x_candidates.append(max(0, W - w))
    y_candidates.append(max(0, H - h))
    z_candidates.append(max(0, D - d))
    
    # Filter valid candidates
    x_candidates = sorted(set([x for x in x_candidates if 0 <= x <= W - w]))
    y_candidates = sorted(set([y for y in y_candidates if 0 <= y <= H - h]))
    z_candidates = sorted(set([z for z in z_candidates if 0 <= z <= D - d]))
    
    if not x_candidates or not y_candidates or not z_candidates:
        return None
    
    best_pos = None
    best_metric = None
    
    for z in z_candidates:
        if max_z is not None and z > max_z:
            continue
        for y in y_candidates:
            if max_y is not None and y > max_y:
                continue
            for x in x_candidates:
                if max_x is not None and x > max_x:
                    continue
                if _fits_at(x, y, z, w, h, d, W, H, D, placed):
                    metric = (z, y, x)
                    if best_metric is None or metric < best_metric:
                        best_pos = (x, y, z)
                        best_metric = metric
                    break  # For this (z,y), we found the best x, move to next y
        # Note: we break x loop but continue y,z loops to find truly lowest
    
    return best_pos

def _pack_first_fit_decreasing(boxes, container, sort_key=None, deadline=None):
    """First-fit decreasing packing with all 6 orientations.
    Returns list of (idx, x, y, z, w, h, d) for placed boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    if sort_key is None:
        sort_key = lambda b: -b[0]*b[1]*b[2]
    
    # Sort boxes
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: sort_key(x[1]))
    
    placed = []  # (idx, x, y, z, w, h, d)
    
    for idx, (ow, oh, od) in indexed_boxes:
        if deadline is not None and time.time() > deadline:
            break
        
        orientations = _all_orientations(ow, oh, od)
        
        best_pos = None
        best_orient = None
        best_metric = None
        
        for (w, h, d) in orientations:
            if w > W or h > H or d > D:
                continue
            
            pos = _find_best_position(w, h, d, W, H, D, placed)
            if pos is not None:
                metric = (pos[2], pos[1], pos[0], w*h*d)  # Prefer lower z,y,x, then larger volume
                if best_metric is None or metric < best_metric:
                    best_pos = pos
                    best_orient = (w, h, d)
                    best_metric = metric
        
        if best_pos is not None:
            x, y, z = best_pos
            w, h, d = best_orient
            placed.append((idx, x, y, z, w, h, d))
    
    return placed

def _pack_shelf(boxes, container, sort_key=None, deadline=None):
    """Shelf-based packing. Returns list of (idx, x, y, z, w, h, d)."""
    W, H, D = container
    n = len(boxes)
    
    if sort_key is None:
        sort_key = lambda b: -b[0]*b[1]*b[2]
    
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: sort_key(x[1]))
    
    placed = []
    
    # Shelf state
    shelf_x = 0
    shelf_y = 0
    shelf_max_h = 0
    layer_z = 0
    layer_max_d = 0
    
    for idx, (ow, oh, od) in indexed_boxes:
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose orientation: minimize height (y-extent) primarily, then footprint
        orientations = _all_orientations(ow, oh, od)
        
        # Pick orientation with smallest y-extent, then smallest max(x,z)
        best_orient = min(orientations, key=lambda o: (o[1], max(o[0], o[2])))
        w, h, d = best_orient
        
        # Try to fit in current shelf
        if shelf_x + w > W:
            # Move to next row
            shelf_y += shelf_max_h
            shelf_x = 0
            shelf_max_h = 0
        
        if shelf_y + h > H:
            # Move to next layer
            layer_z += layer_max_d
            shelf_y = 0
            shelf_x = 0
            shelf_max_h = 0
            layer_max_d = 0
        
        if layer_z + d > D:
            continue
        
        placed.append((idx, shelf_x, shelf_y, layer_z, w, h, d))
        shelf_x += w
        shelf_max_h = max(shelf_max_h, h)
        layer_max_d = max(layer_max_d, d)
    
    return placed

def _pack_layered(boxes, container, sort_key=None, deadline=None):
    """Layer-based packing: fill layers in z direction, each layer is a 2D bin packing in x-y.
    Returns list of (idx, x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    
    if sort_key is None:
        sort_key = lambda b: -b[0]*b[1]*b[2]
    
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: sort_key(x[1]))
    
    placed = []
    
    # Group boxes into layers based on their depth (z-extent)
    # For each box, choose orientation that minimizes z-extent
    box_list = []
    for idx, (ow, oh, od) in indexed_boxes:
        orientations = _all_orientations(ow, oh, od)
        # Pick orientation with smallest z-extent
        best_orient = min(orientations, key=lambda o: (o[2], o[1], o[0]))
        w, h, d = best_orient
        box_list.append((idx, w, h, d))
    
    # Sort by z-extent descending
    box_list.sort(key=lambda x: -x[3])
    
    # Layer packing
    layer_z = 0
    layer_max_d = 0
    
    # Within each layer, do shelf packing in x-y
    while box_list:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find max z-extent among remaining boxes that fit
        fitting = [b for b in box_list if layer_z + b[3] <= D]
        if not fitting:
            break
        
        # Max depth for this layer
        layer_d = max(b[3] for b in fitting)
        
        # Place boxes in this layer using shelf method
        shelf_x = 0
        shelf_y = 0
        shelf_max_h = 0
        layer_boxes = []
        
        remaining = []
        for (idx, w, h, d) in fitting:
            if shelf_x + w > W:
                shelf_y += shelf_max_h
                shelf_x = 0
                shelf_max_h = 0
            
            if shelf_y + h > H:
                # Can't fit in this layer
                remaining.append((idx, w, h, d))
                continue
            
            layer_boxes.append((idx, w, h, d, shelf_x, shelf_y))
            shelf_x += w
            shelf_max_h = max(shelf_max_h, h)
        
        # Add placed boxes
        for (idx, w, h, d, x, y) in layer_boxes:
            placed.append((idx, x, y, layer_z, w, h, d))
        
        layer_z += layer_d
        layer_max_d = layer_d
        box_list = remaining
    
    return placed

def _make_output(placed, n, boxes):
    """Convert placed list to output format.
    placed: list of (idx, x, y, z, w, h, d)
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    output = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for (idx, x, y, z, w, h, d) in placed:
        if 0 <= idx < n:
            output[idx] = (0, x, y, z, w, h, d)
    return output

def _compute_volume(placed):
    """Compute total volume of placed boxes."""
    return sum(w * h * d for (_, _, _, _, w, h, d) in placed)

def _verify(placed, boxes, container):
    """Verify that no boxes overlap and all are within container.
    Returns True if valid.
    """
    W, H, D = container
    n = len(boxes)
    
    for (idx, x, y, z, w, h, d) in placed:
        if idx < 0 or idx >= n:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
    
    # Check overlaps
    for i in range(len(placed)):
        for j in range(i+1, len(placed)):
            _, x1, y1, z1, w1, h1, d1 = placed[i]
            _, x2, y2, z2, w2, h2, d2 = placed[j]
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1):
                return False
    
    return True

def load(boxes, container):
    """
    boxes: list of (w, h, d) tuples
    container: (W, H, D) tuple
    returns: list of (bin, x, y, z, w, h, d) tuples
    """
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try multiple strategies and keep the best valid result
    strategies = []
    
    # Strategy 1: First-fit decreasing with volume sort
    def _str1():
        return _pack_first_fit_decreasing(boxes, container, 
                                           sort_key=lambda b: -b[0]*b[1]*b[2],
                                           deadline=deadline)
    strategies.append(("ffd_vol", _str1))
    
    # Strategy 2: First-fit decreasing with max-dim sort
    def _str2():
        return _pack_first_fit_decreasing(boxes, container,
                                           sort_key=lambda b: -max(b),
                                           deadline=deadline)
    strategies.append(("ffd_maxdim", _str2))
    
    # Strategy 3: Shelf packing with volume sort
    def _str3():
        return _pack_shelf(boxes, container,
                           sort_key=lambda b: -b[0]*b[1]*b[2],
                           deadline=deadline)
    strategies.append(("shelf_vol", _str3))
    
    # Strategy 4: Shelf packing with min-dim sort
    def _str4():
        return _pack_shelf(boxes, container,
                           sort_key=lambda b: -min(b),
                           deadline=deadline)
    strategies.append(("shelf_mindim", _str4))
    
    # Strategy 5: Layered packing
    def _str5():
        return _pack_layered(boxes, container,
                             sort_key=lambda b: -b[0]*b[1]*b[2],
                             deadline=deadline)
    strategies.append(("layered_vol", _str5))
    
    # Strategy 6: FFD with sum sort
    def _str6():
        return _pack_first_fit_decreasing(boxes, container,
                                           sort_key=lambda b: -(b[0]+b[1]+b[2]),
                                           deadline=deadline)
    strategies.append(("ffd_sum", _str6))
    
    # Run all strategies
    best_placed = None
    best_volume = 0
    
    for name, strategy_fn in strategies:
        if time.time() > deadline - 5.0:
            break
        
        try:
            placed = strategy_fn()
            
            # Verify
            if not _verify(placed, boxes, container):
                continue
            
            vol = _compute_volume(placed)
            if vol > best_volume:
                best_volume = vol
                best_placed = placed
        except Exception:
            continue
    
    # If no valid placement found, return all left out
    if best_placed is None:
        return [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return _make_output(best_placed, n, boxes)

# EVOLVE-BLOCK-END
