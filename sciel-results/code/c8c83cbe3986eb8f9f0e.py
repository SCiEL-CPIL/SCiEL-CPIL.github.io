import time
import sys
import heapq
from itertools import product

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: (-x[1][0]*x[1][1]*x[1][2], -max(x[1])))
    
    # Try multiple packing strategies and keep the best
    best_placements = None
    best_score = float('inf')
    
    # Strategy 1: Best-fit decreasing with spatial index
    placements = _best_fit_packing(indexed_boxes, container, start_time, deadline)
    if placements is not None:
        score = _score_placements(placements, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 2: Shelf-based packing with sorting
    placements = _shelf_packing(indexed_boxes, container)
    if placements is not None:
        score = _score_placements(placements, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 3: Layer-based 2D slice decomposition
    if time.time() < deadline - 5.0:
        placements = _layer_packing(indexed_boxes, container, start_time, deadline)
        if placements is not None:
            score = _score_placements(placements, container)
            if score < best_score:
                best_score = score
                best_placements = placements
    
    # Strategy 4: Alternative sort orders
    for sort_key in [_sort_by_height, _sort_by_width, _sort_by_depth]:
        if time.time() >= deadline - 5.0:
            break
        sorted_boxes = sorted(enumerate(boxes), key=sort_key)
        placements = _best_fit_packing(sorted_boxes, container, start_time, deadline)
        if placements is not None:
            score = _score_placements(placements, container)
            if score < best_score:
                best_score = score
                best_placements = placements
    
    if best_placements is None:
        # Fallback: simple shelf
        best_placements = _shelf_packing(indexed_boxes, container)
    
    # Re-index to original order
    result = [None] * n
    for i, (orig_idx, b, x, y, z, w, h, d) in enumerate(best_placements):
        result[orig_idx] = (b, x, y, z, w, h, d)
    
    return result


def _sort_by_volume_desc(x):
    return (-x[1][0]*x[1][1]*x[1][2], -max(x[1]))

def _sort_by_height(x):
    return (-x[1][1], -x[1][0]*x[1][1]*x[1][2])

def _sort_by_width(x):
    return (-x[1][0], -x[1][0]*x[1][1]*x[1][2])

def _sort_by_depth(x):
    return (-x[1][2], -x[1][0]*x[1][1]*x[1][2])


def _score_placements(placements, container):
    if not placements:
        return 0.0
    W, H, D = container
    container_vol = W * H * D
    
    # Count occupied bins
    bin_volumes = {}
    for orig_idx, b, x, y, z, w, h, d in placements:
        if b not in bin_volumes:
            bin_volumes[b] = 0
        bin_volumes[b] += w * h * d
    
    num_bins = len(bin_volumes)
    max_bin = max(bin_volumes.keys())
    last_bin_vol = bin_volumes.get(max_bin, 0)
    last_bin_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
    
    return num_bins + 0.1 * (1 - last_bin_fraction)


def _best_fit_packing(sorted_boxes, container, start_time, deadline):
    W, H, D = container
    n = len(sorted_boxes)
    
    # Each bin maintains a list of placed boxes and free space tracking
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    placements = []
    
    for idx, (orig_idx, w, h, d) in enumerate(sorted_boxes):
        if time.time() > deadline:
            return None
        
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        for b_idx, bin_boxes in enumerate(bins):
            # Find best position in this bin using best-fit
            pos, waste = _find_best_position(bin_boxes, w, h, d, W, H, D)
            if pos is not None:
                if waste < best_waste:
                    best_waste = waste
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Try new bin
            if _can_fit_in_empty(w, h, d, W, H, D):
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            else:
                return None  # Box doesn't fit
        
        b, x, y, z = best_bin, best_pos[0], best_pos[1], best_pos[2]
        bins[b].append((x, y, z, w, h, d))
        placements.append((orig_idx, b, x, y, z, w, h, d))
    
    return placements


def _can_fit_in_empty(w, h, d, W, H, D):
    return w <= W and h <= H and d <= D


def _find_best_position(bin_boxes, w, h, d, W, H, D):
    """Find best-fit position: minimize wasted space above and behind the box."""
    if not bin_boxes:
        if w <= W and h <= H and d <= D:
            return (0, 0, 0), 0.0
    
    # Generate candidate positions from corners of existing boxes
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (bx, by, bz, bw, bh, bd) in bin_boxes:
        # Positions adjacent to each face
        candidates.add((bx + bw, by, bz))
        candidates.add((bx, by + bh, bz))
        candidates.add((bx, by, bz + bd))
        # Also try some combinations
        candidates.add((bx + bw, by + bh, bz))
        candidates.add((bx + bw, by, bz + bd))
        candidates.add((bx, by + bh, bz + bd))
        candidates.add((bx + bw, by + bh, bz + bd))
    
    best_pos = None
    best_score = float('inf')
    
    for (cx, cy, cz) in candidates:
        if cx + w > W or cy + h > H or cz + d > D:
            continue
        
        # Check overlap
        if _overlaps_any((cx, cy, cz, w, h, d), bin_boxes):
            continue
        
        # Score: prefer lower z, then lower y, then lower x (to fill from bottom-left-front)
        # Use a composite score that favors compactness
        score = cz * 10000 + cy * 100 + cx
        # Also consider how well it fills: minimize the "gap" above
        if score < best_score:
            best_score = score
            best_pos = (cx, cy, cz)
    
    return best_pos, best_score if best_pos is not None else float('inf')


def _overlaps_any(box, bin_boxes):
    bx, by, bz, bw, bh, bd = box
    for (ox, oy, oz, ow, oh, od) in bin_boxes:
        if bx < ox + ow and bx + bw > ox and \
           by < oy + oh and by + bh > oy and \
           bz < oz + od and bz + bd > oz:
            return True
    return False


def _shelf_packing(sorted_boxes, container):
    W, H, D = container
    n = len(sorted_boxes)
    
    bins = []  # each bin: list of (x, y, z, w, h, d), plus shelf tracking
    # For each bin, track: current shelf y, current row x, shelf height, layer z, layer depth
    
    placements = []
    
    for orig_idx, w, h, d in sorted_boxes:
        placed = False
        
        for b_idx in range(len(bins)):
            # Get state for this bin
            bin_data = bins[b_idx]
            boxes_in_bin = bin_data['boxes']
            shelf_y = bin_data['shelf_y']
            row_x = bin_data['row_x']
            shelf_h = bin_data['shelf_h']
            layer_z = bin_data['layer_z']
            layer_d = bin_data['layer_d']
            
            # Try to place in current shelf
            if row_x + w <= W:
                x, y, z = row_x, shelf_y, layer_z
            else:
                # New row in same shelf
                row_x = 0
                shelf_y += shelf_h
                shelf_h = 0
                if shelf_y + h <= H:
                    x, y, z = 0, shelf_y, layer_z
                else:
                    # New layer
                    layer_z += layer_d
                    layer_d = 0
                    shelf_y = 0
                    shelf_h = 0
                    if layer_z + d <= D:
                        x, y, z = 0, 0, layer_z
                    else:
                        continue  # Can't fit in this bin
            
            # Check overlap with existing boxes
            if not _overlaps_any((x, y, z, w, h, d), boxes_in_bin):
                boxes_in_bin.append((x, y, z, w, h, d))
                row_x = x + w
                shelf_h = max(shelf_h, h)
                layer_d = max(layer_d, d)
                placements.append((orig_idx, b_idx, x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # New bin
            b_idx = len(bins)
            bins.append({
                'boxes': [],
                'shelf_y': 0,
                'row_x': 0,
                'shelf_h': 0,
                'layer_z': 0,
                'layer_d': 0
            })
            x, y, z = 0, 0, 0
            bins[b_idx]['boxes'].append((x, y, z, w, h, d))
            bins[b_idx]['row_x'] = w
            bins[b_idx]['shelf_h'] = h
            bins[b_idx]['layer_d'] = d
            placements.append((orig_idx, b_idx, x, y, z, w, h, d))
    
    return placements


def _layer_packing(sorted_boxes, container, start_time, deadline):
    """Layer-based: group boxes by similar depth, pack each layer as 2D."""
    W, H, D = container
    n = len(sorted_boxes)
    
    if not sorted_boxes:
        return []
    
    # Group boxes into layers by depth
    # Sort by depth, then assign to layers
    boxes_with_idx = [(orig_idx, w, h, d) for orig_idx, w, h, d in sorted_boxes]
    
    # Simple approach: process boxes in order, assign to layers
    layers = []  # each layer: list of (orig_idx, w, h, d), z_offset, depth
    
    for orig_idx, w, h, d in boxes_with_idx:
        if time.time() > deadline:
            return None
        
        placed = False
        for layer_idx, layer in enumerate(layers):
            if layer['z'] + d <= D:
                # Try to fit in this 2D layer
                if _fits_in_2d_layer(layer, w, h, W, H):
                    layer['boxes'].append((orig_idx, w, h, d))
                    layer['depth'] = max(layer['depth'], d)
                    placed = True
                    break
        
        if not placed:
            # New layer
            if not layers:
                z = 0
            else:
                z = layers[-1]['z'] + layers[-1]['depth']
            if z + d <= D:
                layers.append({
                    'boxes': [(orig_idx, w, h, d)],
                    'z': z,
                    'depth': d
                })
            else:
                return None  # Can't fit
    
    # Now pack each 2D layer
    placements = []
    bin_idx = 0
    
    for layer in layers:
        # Pack the 2D boxes in this layer using shelf method
        layer_boxes = layer['boxes']
        z = layer['z']
        
        # 2D shelf packing
        row_x = 0
        shelf_y = 0
        shelf_h = 0
        
        for orig_idx, w, h, d in layer_boxes:
            if row_x + w > W:
                row_x = 0
                shelf_y += shelf_h
                shelf_h = 0
            
            if shelf_y + h > H:
                # Can't fit in this layer's height, need new bin
                # This is a simplification; in practice we'd need to split
                break
            
            x, y = row_x, shelf_y
            placements.append((orig_idx, bin_idx, x, y, z, w, h, d))
            row_x += w
            shelf_h = max(shelf_h, h)
    
    return placements if placements else None


def _fits_in_2d_layer(layer, w, h, W, H):
    """Check if a box fits in the 2D layer (simplified check)."""
    return w <= W and h <= H
