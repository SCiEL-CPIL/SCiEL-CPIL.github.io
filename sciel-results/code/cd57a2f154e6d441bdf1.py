import time
import math
import random
from typing import List, Tuple
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: Evaluator kills at 150s, task says 120s. Use 115s to be safe.
    deadline = time.time() + 115.0
    
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    min_bins = max(1, math.ceil(total_vol / container_vol))
    
    # Sort indices by volume descending (Largest First)
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def compute_score(placements, W, H, D):
        if not placements:
            return 0.0
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_bin_vol = sum(p[3] * p[4] * p[5] for p in placements if p[0] == last_bin)
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def find_positions_skyline(existing_items, w, h, d, W, H, D):
        """
        Find valid positions using a 2D Skyline (Column) Representation.
        This projects the 3D problem onto the XY plane.
        We maintain a set of 'skyline' rectangles that represent the highest z-coordinate
        reachable at any point in the XY plane, given the existing items.
        
        Actually, a simpler and more robust approach for "Skyline" in this context:
        1. Collect all unique x and y coordinates from existing items (0, ex, ex+ew, ey, ey+eh).
        2. This creates a grid of cells in the XY plane.
        3. For each cell, calculate the minimum z-height required to place a box there.
           This is determined by the maximum top-face (z+ed) of any existing item that
           overlaps with the cell in the XY plane.
        4. A box (w,h,d) can be placed at (cx, cy, cz) if:
           - The rectangle [cx, cx+w] x [cy, cy+h] fits within the container.
           - For all cells covered by this rectangle, the calculated skyline height + d <= D.
           - AND the box doesn't overlap in 3D (which is guaranteed if we pick cz = max_skyline_height).
        
        Wait, the "Skyline" method usually implies we place the box at the lowest possible z.
        So for a given (cx, cy), the lowest valid z is the max of the top-faces of all items
        overlapping the XY footprint of the new box.
        
        Let's implement this:
        1. Generate candidate (cx, cy) pairs from the grid defined by item edges.
        2. For each candidate (cx, cy), check if the footprint fits in W x H.
        3. Calculate required_z = max(z_i + d_i) for all items i that overlap the footprint in XY.
           If no overlap, required_z = 0.
        4. If required_z + d <= D, then (cx, cy, required_z) is a valid position.
        5. We want to minimize z, then y, then x (Bottom-Left-Back).
        """
        if not existing_items:
            if w <= W and h <= H and d <= D:
                return [(0, 0, 0)]
            return []
        
        # Collect unique x and y coordinates
        x_coords = {0, W}
        y_coords = {0, H}
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
        
        x_coords = sorted([x for x in x_coords if x <= W])
        y_coords = sorted([y for y in y_coords if y <= H])
        
        valid_positions = []
        
        # Iterate over potential bottom-left corners (cx, cy)
        # cx must be such that cx + w <= W
        # cy must be such that cy + h <= H
        for cx in x_coords:
            if cx + w > W:
                continue
            for cy in y_coords:
                if cy + h > H:
                    continue
                
                # Calculate the minimum z required for this footprint
                # This is the max top-face of any item that overlaps [cx, cx+w] x [cy, cy+h]
                min_z = 0
                for (ex, ey, ez, ew, eh, ed) in existing_items:
                    # Check XY overlap
                    if cx < ex + ew and ex < cx + w and cy < ey + eh and ey < cy + h:
                        top_z = ez + ed
                        if top_z > min_z:
                            min_z = top_z
                
                if min_z + d <= D:
                    valid_positions.append((cx, cy, min_z))
        
        # Sort by z, then y, then x to find the "best" (lowest, most compact) position
        valid_positions.sort(key=lambda p: (p[2], p[1], p[0]))
        
        return valid_positions

    def solve(ordering):
        """
        Run the packing algorithm with a specific ordering of box indices.
        """
        state = {} # bin_id -> list of (x,y,z,w,h,d)
        placements = [None] * len(ordering)
        
        for i, box_idx in enumerate(ordering):
            if time.time() > deadline:
                return None
            
            w, h, d = boxes[box_idx]
            
            best_bin = None
            best_pos = None
            best_score = float('inf')
            
            # Try existing bins
            for bin_id in sorted(state.keys()):
                pos_list = find_positions_skyline(state[bin_id], w, h, d, W, H, D)
                if not pos_list:
                    continue
                
                # The first position in the sorted list is the best for this bin
                # (lowest z, then y, then x)
                pos = pos_list[0]
                
                # We need a global comparison across bins.
                # Heuristic: Prefer bins where the box fits "lower" and "more compact".
                # Since we want to minimize bins, we should prefer filling existing bins.
                # Among existing bins, prefer the one where the box is placed at the lowest z/y/x.
                
                # Score: z * (H*D) + y * D + x
                s = pos[2] * (H * D) + pos[1] * D + pos[0]
                
                if s < best_score:
                    best_score = s
                    best_bin = bin_id
                    best_pos = pos
                        
            # If no existing bin works, open a new one
            if best_bin is None:
                new_bin_id = max(state.keys()) + 1 if state else 0
                best_bin = new_bin_id
                best_pos = (0, 0, 0)
            
            # Place the box
            if best_bin not in state:
                state[best_bin] = []
            state[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
            placements[i] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            
        # Reorder placements to match original box index order
        final_placements = [None] * n
        for i, box_idx in enumerate(ordering):
            final_placements[box_idx] = placements[i]
            
        return final_placements

    def local_search_polish(placements, max_iterations=50):
        """
        Local search: try to move boxes to other bins to reduce total bins used
        or improve last-bin fill.
        """
        if not placements:
            return placements
        
        # Group boxes by bin
        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)
        
        current_score = compute_score(placements, W, H, D)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Try moving each box to another bin if it fits
            # Process larger boxes first as they are harder to place
            for box_idx in sorted(range(n), key=lambda i: -box_vols[i]):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                
                # Try other bins
                other_bins = [bid for bid in bin_boxes if bid != b]
                for target_bin in other_bins:
                    # Check if box fits in target bin
                    target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_boxes[target_bin]]
                    pos_list = find_positions_skyline(target_items, w, h, d, W, H, D)
                    if pos_list:
                        pos = pos_list[0]
                        # Tentatively move
                        new_placements = list(placements)
                        new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements, W, H, D)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            # Update bin groupings
                            bin_boxes[b].remove(box_idx)
                            if not bin_boxes[b]:
                                del bin_boxes[b]
                            bin_boxes[target_bin].append(box_idx)
                            improved = True
                            break
        
        return placements

    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Volume Descending (Standard BFD with Skyline)
    result = solve(indices_sorted)
    if result:
        s = compute_score(result, W, H, D)
        if s < best_score:
            best_score = s
            best_result = result
            # Apply local search polish
            polished = local_search_polish(best_result)
            ps = compute_score(polished, W, H, D)
            if ps < best_score:
                best_score = ps
                best_result = polished

    # Strategy 2: Volume Descending, but tie-break by max dimension
    indices_sorted2 = sorted(indices, key=lambda i: (box_vols[i], max(boxes[i])), reverse=True)
    if time.time() < deadline - 10:
        result = solve(indices_sorted2)
        if result:
            s = compute_score(result, W, H, D)
            if s < best_score:
                best_score = s
                best_result = result
                polished = local_search_polish(best_result)
                ps = compute_score(polished, W, H, D)
                if ps < best_score:
                    best_score = ps
                    best_result = polished

    # Strategy 3: Randomized Restart with small perturbation
    if time.time() < deadline - 15:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5:
            # Create a slightly perturbed ordering
            perturbed = indices_sorted[:]
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, len(perturbed) - 2)
                perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            result = solve(perturbed)
            if result:
                s = compute_score(result, W, H, D)
                if s < best_score:
                    best_score = s
                    best_result = result
                    polished = local_search_polish(best_result)
                    ps = compute_score(polished, W, H, D)
                    if ps < best_score:
                        best_score = ps
                        best_result = polished
            attempts += 1

    # Fallback if something went wrong (should not happen)
    if best_result is None:
        # Simple shelf as absolute last resort
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

    return best_result
