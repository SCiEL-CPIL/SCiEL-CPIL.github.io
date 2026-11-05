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
    
    deadline = time.time() + 115.0  # safety margin before 120s (evaluator kills at 150s per spec, but task says 120s)
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    min_bins = max(1, math.ceil(total_vol / container_vol))
    
    # Sort boxes by volume descending (larger first) for better heuristics
    # But we must return placements in original order, so track indices
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)
    
    def find_valid_positions(existing_items, w, h, d, W, H, D, max_candidates=200):
        """
        Find all valid positions for a box in a bin.
        Returns list of (cx, cy, cz) positions.
        """
        # Candidate positions: (0,0,0) and all 8 corner offsets from each existing item
        candidates = [(0, 0, 0)]
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            for dx in [0, ew]:
                for dy in [0, eh]:
                    for dz in [0, ed]:
                        candidates.append((ex + dx, ey + dy, ez + dz))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Sort candidates by (z, y, x) to prefer lower positions
        unique_candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        valid_positions = []
        for (cx, cy, cz) in unique_candidates[:max_candidates]:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            # Check overlap with all existing items
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if check_overlap(cx, cy, cz, w, h, d, ex, ey, ez, ew, eh, ed):
                    overlap = True
                    break
            if not overlap:
                valid_positions.append((cx, cy, cz))
        
        return valid_positions
    
    def evaluate_position(existing_items, w, h, d, pos, W, H, D):
        """
        Score a position: lower is better.
        Prefer positions that fill the bin more densely (minimize wasted space).
        """
        cx, cy, cz = pos
        # Heuristic: prefer positions closer to origin (lower x, y, z)
        score = cx + cy + cz
        
        # Penalize if this creates gaps: check how much free space is left
        # in each dimension after placing this box
        # Track max extent in each direction
        max_x = cx + w
        max_y = cy + h
        max_z = cz + d
        
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            max_x = max(max_x, ex + ew)
            max_y = max(max_y, ey + eh)
            max_z = max(max_z, ez + ed)
        
        # Penalize large gaps: if the box is placed far from origin, 
        # it may leave empty space that can't be filled
        gap_penalty = 0
        if cx > 0:
            # Check if there's space to the left that can't be filled
            gap_penalty += cx * 0.1
        if cy > 0:
            gap_penalty += cy * 0.1
        if cz > 0:
            gap_penalty += cz * 0.1
        
        return score + gap_penalty
    
    def try_place(placement_state, box_idx):
        """
        placement_state: dict mapping bin_id -> list of (x,y,z,w,h,d)
        Try to place box_idx in existing bins first (best fit), then new bin.
        Returns (success, new_state, bin_id, pos)
        """
        w, h, d = boxes[box_idx]
        
        # Try existing bins
        best_bin = None
        best_pos = None
        best_score = float('inf')
        
        for bin_id in sorted(placement_state.keys()):
            if time.time() > deadline:
                break
            # Try to find positions in this bin
            valid_positions = find_valid_positions(placement_state[bin_id], w, h, d, W, H, D)
            if valid_positions:
                # Evaluate each valid position and pick the best
                for pos in valid_positions[:10]:  # Limit to first 10 for speed
                    score = evaluate_position(placement_state[bin_id], w, h, d, pos, W, H, D)
                    if score < best_score:
                        best_score = score
                        best_bin = bin_id
                        best_pos = pos
        
        if best_bin is not None:
            new_bin_list = placement_state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
            new_state = dict(placement_state)
            new_state[best_bin] = new_bin_list
            return True, new_state, best_bin, best_pos
        
        # Try new bin
        new_bin_id = max(placement_state.keys()) + 1 if placement_state else 0
        new_state = dict(placement_state)
        new_state[new_bin_id] = [(0, 0, 0, w, h, d)]
        return True, new_state, new_bin_id, (0, 0, 0)
    
    def local_improve(placements, W, H, D, deadline):
        """
        Try to improve the placement by moving boxes to reduce bin count
        or improve last-bin fill.
        """
        if any(p is None for p in placements):
            return placements
        
        # Group boxes by bin
        bin_groups = {}
        for i, (bin_id, x, y, z, w, h, d) in enumerate(placements):
            if bin_id not in bin_groups:
                bin_groups[bin_id] = []
            bin_groups[bin_id].append(i)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to move each box to an earlier bin if possible
            for i in range(n):
                if time.time() > deadline:
                    break
                bin_id, x, y, z, w, h, d = placements[i]
                # Try to move to each lower-numbered bin
                for target_bin in sorted(bin_groups.keys()):
                    if target_bin >= bin_id:
                        break
                    # Check if box fits in target_bin
                    target_items = [(placements[j][1], placements[j][2], placements[j][3], 
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_groups[target_bin]]
                    valid_positions = find_valid_positions(target_items, w, h, d, W, H, D, max_candidates=50)
                    if valid_positions:
                        # Move the box
                        new_pos = valid_positions[0]
                        placements[i] = (target_bin, new_pos[0], new_pos[1], new_pos[2], w, h, d)
                        bin_groups[target_bin].append(i)
                        if i in bin_groups[bin_id]:
                            bin_groups[bin_id].remove(i)
                        if not bin_groups[bin_id]:
                            del bin_groups[bin_id]
                        improved = True
                        break
        
        # Renumber bins to be consecutive starting from 0
        if bin_groups:
            sorted_bins = sorted(bin_groups.keys())
            bin_map = {old: new for new, old in enumerate(sorted_bins)}
            for i in range(n):
                bin_id, x, y, z, w, h, d = placements[i]
                placements[i] = (bin_map[bin_id], x, y, z, w, h, d)
        
        return placements
    
    # Main optimization loop
    best_result = None
    best_score = float('inf')
    
    # Multiple passes with different orderings
    orderings = [
        indices_sorted,  # volume descending
        list(reversed(indices_sorted)),  # volume ascending
        indices,  # original order
        sorted(indices, key=lambda i: boxes[i][0] * boxes[i][1], reverse=True),  # base area
        sorted(indices, key=lambda i: max(boxes[i]), reverse=True),  # max dimension
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        state = {}
        placements = [None] * n
        
        for box_idx in ordering:
            if time.time() > deadline:
                break
            success, state, bin_id, pos = try_place(state, box_idx)
            if success:
                placements[box_idx] = (bin_id, pos[0], pos[1], pos[2], boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
        
        # Check if all placed
        if all(p is not None for p in placements):
            # Apply local improvement
            placements = local_improve(placements, W, H, D, deadline)
            score = compute_score(placements, W, H, D)
            if score < best_score:
                best_score = score
                best_result = placements
    
    # If no valid result found, fall back to simple shelf
    if best_result is None or any(p is None for p in best_result):
        # Simple shelf heuristic as fallback
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


def compute_score(placements, W, H, D):
    """
    Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume)
    where B is the number of occupied bins (per instance, but we have one instance here)
    Actually, the score is defined on the whole instance: B is the number of occupied bins.
    """
    if not placements:
        return 0.0
    
    # Find all bins used
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    
    # Find the last bin (largest label)
    last_bin = max(bins_used)
    
    # Compute volume occupied in the last bin
    container_vol = W * H * D
    last_bin_vol = sum(p[3] * p[4] * p[5] for p in placements if p[0] == last_bin)
    
    # Score
    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
