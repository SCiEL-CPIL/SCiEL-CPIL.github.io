"""Rewritten 3D bin packing with First-Fit-Decreasing + best-position search."""
import time
import math

def _sort_boxes(boxes):
    # Sort by volume descending, then by max dimension descending for tie-breaking
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-t[1][0] * t[1][1] * t[1][2], -max(t[1])))
    return indexed

def _can_place(placed, x, y, z, w, h, d, W, H, D):
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap in 3D
        if (x < px + pw and px < x + w and
            y < py + ph and py < y + h and
            z < pz + pd and pz < z + d):
            return False
    return True

def _find_best_position(placed, box, W, H, D):
    """Find the best position for a box in a bin using a scoring heuristic.
    Lower score is better. Returns (x, y, z) or None if no position found."""
    w, h, d = box
    best_pos = None
    best_score = float('inf')
    
    # Generate candidate positions from existing boxes plus origin
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # Place adjacent in +x, +y, +z directions
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
    
    # Also try some grid-based positions for better coverage
    # But limit to keep it fast
    seen = set()
    for (cx, cy, cz) in candidates:
        if (cx, cy, cz) in seen:
            continue
        seen.add((cx, cy, cz))
        if _can_place(placed, cx, cy, cz, w, h, d, W, H, D):
            # Score: prefer lower x, then lower y, then lower z (to fill compactly)
            # Use a weighted sum to encourage packing towards origin
            score = cx * 1.0 + cy * 0.8 + cz * 0.6
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
    
    return best_pos

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (First-Fit-Decreasing)
    sorted_boxes = _sort_boxes(boxes)
    
    # bins[i] = list of (x, y, z, w, h, d) placements in bin i
    bins = []
    # box_order[i] = original index of the i-th box in sorted order
    box_order = [idx for idx, _ in sorted_boxes]
    
    # Placement results: for each original box index, store (bin, x, y, z, w, h, d)
    result = [None] * n
    
    for si in range(n):
        if time.time() > deadline:
            break
        
        orig_idx = box_order[si]
        w, h, d = boxes[orig_idx]
        
        placed_in_bin = -1
        placed_pos = None
        
        for b in range(len(bins)):
            if time.time() > deadline:
                break
            pos = _find_best_position(bins[b], (w, h, d), W, H, D)
            if pos is not None:
                placed_in_bin = b
                placed_pos = pos
                break
        
        if placed_in_bin == -1:
            # Try to place in a new bin
            if _can_place([], 0, 0, 0, w, h, d, W, H, D):
                placed_in_bin = len(bins)
                bins.append([])
                placed_pos = (0, 0, 0)
        
        if placed_in_bin == -1:
            # Box doesn't fit in any bin even empty - shouldn't happen if box fits container
            # Place in a new bin at origin anyway (may be invalid but avoids crash)
            placed_in_bin = len(bins)
            bins.append([])
            placed_pos = (0, 0, 0)
        
        x, y, z = placed_pos
        bins[placed_in_bin].append((x, y, z, w, h, d))
        result[orig_idx] = (placed_in_bin, x, y, z, w, h, d)
    
    return result
