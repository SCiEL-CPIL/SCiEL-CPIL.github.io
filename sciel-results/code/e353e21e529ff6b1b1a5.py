"""Model-and-optimize 3D bin packing with deadline guard."""
import time
import random
import math
from typing import List, Tuple

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    
    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def find_position(existing_items, w, h, d, W, H, D):
        """
        Find a position (x, y, z) where box (w, h, d) fits without overlap.
        Uses all unique coordinate combinations from box faces.
        """
        if not existing_items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Collect unique x, y, z coordinates from box faces
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
            z_coords.add(ez)
            z_coords.add(ez + ed)
        
        # Filter to only coordinates where the box would fit
        x_coords = sorted(x for x in x_coords if x + w <= W)
        y_coords = sorted(y for y in y_coords if y + h <= H)
        z_coords = sorted(z for z in z_coords if z + d <= D)
        
        if not x_coords or not y_coords or not z_coords:
            return None
        
        # Generate candidate positions and sort by (z, y, x) - bottom-left-front
        candidates = []
        for cx in x_coords:
            for cy in y_coords:
                for cz in z_coords:
                    candidates.append((cx, cy, cz))
        
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        for (cx, cy, cz) in candidates:
            # Check overlap with all existing items
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                return (cx, cy, cz)
        
        return None
    
    def try_place_in_bin(bin_items, w, h, d, W, H, D):
        """Try to place box in a specific bin, return position or None."""
        return find_position(bin_items, w, h, d, W, H, D)
    
    def pack_with_order(ordering, state_init=None):
        """Pack boxes in given order. Returns placements list or None."""
        state = {} if state_init is None else dict(state_init)
        placements = [None] * n
        all_placed = True
        
        for box_idx in ordering:
            if time.time() > deadline:
                all_placed = False
                break
            
            w, h, d = boxes[box_idx]
            
            # Try existing bins: prefer bin that minimizes wasted space
            # Strategy: try bins sorted by current volume descending (fill fuller bins first)
            best_bin = None
            best_pos = None
            best_waste = float('inf')
            
            bin_ids = list(state.keys())
            # Sort bins by current fill volume descending
            bin_vols = {bid: sum(ew*eh*ed for (_,_,_,ew,eh,ed) in state[bid]) for bid in bin_ids}
            bin_ids_sorted = sorted(bin_ids, key=lambda bid: bin_vols[bid], reverse=True)
            
            for bin_id in bin_ids_sorted:
                pos = try_place_in_bin(state[bin_id], w, h, d, W, H, D)
                if pos is not None:
                    # Compute waste: how much empty space remains in this bin
                    # Prefer bins where adding this box leaves least unused capacity
                    current_vol = bin_vols[bin_id]
                    new_vol = current_vol + w * h * d
                    waste = container_vol - new_vol
                    # Also consider: prefer tighter fit (smaller gap to container volume)
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = bin_id
                        best_pos = pos
            
            if best_bin is not None:
                state[best_bin] = state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                # Open new bin
                new_bin_id = max(state.keys()) + 1 if state else 0
                state[new_bin_id] = [(0, 0, 0, w, h, d)]
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)
        
        if all_placed and all(p is not None for p in placements):
            return placements
        return None
    
    # Main optimization: try multiple orderings
    best_result = None
    best_score = float('inf')
    
    # Generate diverse orderings
    orderings = []
    
    # 1. Volume descending
    orderings.append(indices_sorted)
    
    # 2. Volume ascending
    orderings.append(list(reversed(indices_sorted)))
    
    # 3. Original order
    orderings.append(indices)
    
    # 4. Max dimension descending
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 5. Surface area descending
    orderings.append(sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 6. Aspect ratio based
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1], boxes[i][2]) / max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))
    
    # 7. Randomized volume descending (two variants)
    for seed in [42, 123]:
        random.seed(seed)
        shuffled = indices_sorted[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    # 8. Weighted by volume^0.5
    orderings.append(sorted(indices, key=lambda i: math.sqrt(box_vols[i]), reverse=True))
    
    # 9. Sort by minimum dimension
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    
    # 10. Sort by maximum dimension
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 11. Sort by (w+h+d) descending - larger perimeter first
    orderings.append(sorted(indices, key=lambda i: boxes[i][0]+boxes[i][1]+boxes[i][2], reverse=True))
    
    # 12. Sort by min(w,h) descending - wider bases first
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1]), reverse=True))
    
    # Run all orderings and keep the best
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        result = pack_with_order(ordering)
        if result is not None:
            score = compute_score(result, W, H, D)
            if score < best_score:
                best_score = score
                best_result = result
    
    # If no valid result found, fall back to simple shelf
    if best_result is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements
    
    # Post-process: relabel bins to be consecutive starting from 0
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]
    
    return best_result


def compute_score(placements, W, H, D):
    if not placements:
        return 0.0
    
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    
    last_bin = max(bins_used)
    
    container_vol = W * H * D
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    
    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
