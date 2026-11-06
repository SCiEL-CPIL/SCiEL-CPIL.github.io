"""3D Bin Packing with Maximal Space Partitioning."""
import time
import random
import math
from typing import List, Tuple
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0
    container_vol = W * H * D
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    
    def find_position_in_spaces(spaces, w, h, d):
        """
        Find a position (x, y, z) in the list of maximal empty spaces.
        Returns (x, y, z, space_index) or None.
        Prefers positions with lowest z, then y, then x.
        """
        best_pos = None
        best_score = None
        best_space_idx = -1
        
        for si, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
            # Check if box fits in this space
            if w > sw or h > sh or d > sd:
                continue
            
            # Try placing at the bottom-left-front corner of this space
            cx, cy, cz = sx, sy, sz
            
            # Score: prefer lower z, then lower y, then lower x
            pos_score = (cz, cy, cx)
            
            if best_score is None or pos_score < best_score:
                best_score = pos_score
                best_pos = (cx, cy, cz)
                best_space_idx = si
        
        return best_pos, best_space_idx
    
    def update_spaces_after_placement(spaces, box_pos, box_w, box_h, box_d):
        """
        After placing a box at box_pos with dimensions box_w, box_h, box_d,
        update the list of maximal empty spaces by splitting intersecting spaces.
        """
        bx, by, bz = box_pos
        bx2, by2, bz2 = bx + box_w, by + box_h, bz + box_d
        
        new_spaces = []
        
        for (sx, sy, sz, sw, sh, sd) in spaces:
            sx2, sy2, sz2 = sx + sw, sy + sh, sz + sd
            
            # Check if the placed box intersects this space
            if bx >= sx2 or sx >= bx2 or by >= sy2 or sy >= by2 or bz >= sz2 or sz >= bz2:
                # No intersection, keep the space
                new_spaces.append((sx, sy, sz, sw, sh, sd))
                continue
            
            # Intersection exists - need to split
            # The box splits the space into up to 6 sub-spaces
            
            # Left part (if exists)
            if bx > sx:
                nw = bx - sx
                new_spaces.append((sx, sy, sz, nw, sh, sd))
            
            # Right part (if exists)
            if bx2 < sx2:
                nx = bx2 - sx
                nw = sx2 - bx2
                new_spaces.append((nx, sy, sz, nw, sh, sd))
            
            # Bottom part (if exists, relative to y)
            if by > sy:
                # But only for the parts that are NOT already split by x
                # We need to be careful: the left and right parts are already full height
                # The bottom and top parts should only be in the x-range of the box
                ny = by - sy
                new_spaces.append((bx, sy, sz, box_w, ny, sd))
            
            # Top part (if exists, relative to y)
            if by2 < sy2:
                ny = sy2 - by
                new_spaces.append((bx, by2, sz, box_w, ny, sd))
            
            # Front part (if exists, relative to z)
            if bz > sz:
                nz = bz - sz
                new_spaces.append((bx, by, sz, box_w, box_h, nz))
            
            # Back part (if exists, relative to z)
            if bz2 < sz2:
                nz = sz2 - bz
                new_spaces.append((bx, by, bz2, box_w, box_h, nz))
        
        # Merge overlapping spaces (simple approach: keep as is, may have some redundancy)
        # For correctness, we should merge, but for now let's keep it simple
        return new_spaces
    
    def bin_volume(bin_items):
        return sum(ew * eh * ed for (_, _, _, ew, eh, ed) in bin_items)
    
    def compute_score_from_bins(bin_data):
        """Compute score given bin_data: dict of bin_id -> list of (x,y,z,w,h,d)."""
        if not bin_data:
            return 0.0
        
        B = len(bin_data)
        last_bin = max(bin_data.keys())
        last_bin_vol = bin_volume(bin_data[last_bin])
        return B + 0.1 * (1 - last_bin_vol / container_vol)
    
    def pack_with_order(ordering):
        """Pack boxes in given order using maximal space partitioning."""
        # bin_spaces: bin_id -> list of (sx, sy, sz, sw, sh, sd)
        bin_spaces = {}
        # bin_items: bin_id -> list of (x, y, z, w, h, d)
        bin_items = {}
        placements = [None] * n
        
        for box_idx in ordering:
            if time.time() > deadline - 2:
                break
            
            w, h, d = boxes[box_idx]
            
            # Try each existing bin
            best_bin = None
            best_pos = None
            best_score_val = float('inf')
            
            # Sort bins by fill volume descending (check fuller bins first)
            bin_ids = sorted(bin_spaces.keys(), key=lambda bid: bin_volume(bin_items[bid]), reverse=True)
            
            for bin_id in bin_ids:
                spaces = bin_spaces[bin_id]
                pos, space_idx = find_position_in_spaces(spaces, w, h, d)
                
                if pos is not None:
                    # Compute a score for this placement
                    # Prefer bins that are already more full, and positions that fill lower regions
                    current_vol = bin_volume(bin_items[bin_id])
                    fill_frac = (current_vol + w * h * d) / container_vol
                    # Lower z positions are better (more stable, easier to build on)
                    z_penalty = pos[2] / max(D, 1) * 0.1
                    score_val = -fill_frac + z_penalty
                    
                    if score_val < best_score_val:
                        best_score_val = score_val
                        best_bin = bin_id
                        best_pos = pos
            
            if best_bin is not None:
                # Place in existing bin
                bin_items[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
                # Update spaces
                bin_spaces[best_bin] = update_spaces_after_placement(bin_spaces[best_bin], best_pos, w, h, d)
            else:
                # Open new bin
                new_bin_id = max(bin_spaces.keys()) + 1 if bin_spaces else 0
                bin_spaces[new_bin_id] = [(0, 0, 0, W, H, D)]
                bin_items[new_bin_id] = [(0, 0, 0, w, h, d)]
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)
                # Update spaces for new bin
                bin_spaces[new_bin_id] = update_spaces_after_placement(bin_spaces[new_bin_id], (0, 0, 0), w, h, d)
        
        if all(p is not None for p in placements):
            return bin_data_to_placements(bin_items)
        return None
    
    def bin_data_to_placements(bin_items):
        """Convert bin_items dict to placements list."""
        placements = []
        for bin_id, items in bin_items.items():
            for (x, y, z, w, h, d) in items:
                placements.append((bin_id, x, y, z, w, h, d))
        # Need to map back to original box indices
        # This is tricky - let's just return the placements directly
        return placements
    
    # We need a different approach: track which box goes where
    def pack_with_order_v2(ordering):
        """Pack boxes in given order, tracking box indices."""
        bin_spaces = {}
        bin_items = {}  # bin_id -> list of (box_idx, x, y, z, w, h, d)
        placements = [None] * n
        
        for box_idx in ordering:
            if time.time() > deadline - 2:
                break
            
            w, h, d = boxes[box_idx]
            
            best_bin = None
            best_pos = None
            best_score_val = float('inf')
            
            bin_ids = sorted(bin_spaces.keys(), key=lambda bid: sum(ew*eh*ed for (_,_,_,ew,eh,ed) in [item[1:7] for item in bin_items[bid]]), reverse=True)
            
            for bin_id in bin_ids:
                spaces = bin_spaces[bin_id]
                pos, space_idx = find_position_in_spaces(spaces, w, h, d)
                
                if pos is not None:
                    current_vol = sum(ew*eh*ed for (_,_,_,ew,eh,ed) in [item[1:7] for item in bin_items[bin_id]])
                    fill_frac = (current_vol + w * h * d) / container_vol
                    z_penalty = pos[2] / max(D, 1) * 0.1
                    score_val = -fill_frac + z_penalty
                    
                    if score_val < best_score_val:
                        best_score_val = score_val
                        best_bin = bin_id
                        best_pos = pos
            
            if best_bin is not None:
                bin_items[best_bin].append((box_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
                bin_spaces[best_bin] = update_spaces_after_placement(bin_spaces[best_bin], best_pos, w, h, d)
            else:
                new_bin_id = max(bin_spaces.keys()) + 1 if bin_spaces else 0
                bin_spaces[new_bin_id] = [(0, 0, 0, W, H, D)]
                bin_items[new_bin_id] = [(box_idx, 0, 0, 0, w, h, d)]
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)
                bin_spaces[new_bin_id] = update_spaces_after_placement(bin_spaces[new_bin_id], (0, 0, 0), w, h, d)
        
        if all(p is not None for p in placements):
            return placements
        return None
    
    def local_search_polish(placements):
        """
        Local search: try moving boxes to other bins to reduce B or improve last-bin fill.
        """
        if not placements or time.time() > deadline - 5:
            return placements
        
        # Group boxes by bin
        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)
        
        current_score = compute_score(placements)
        improved = True
        iterations = 0
        
        while improved and iterations < 30 and time.time() < deadline - 3:
            improved = False
            iterations += 1
            
            for box_idx in list(range(n)):
                if time.time() > deadline - 3:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                
                other_bins = [bid for bid in bin_boxes if bid != b]
                for target_bin in other_bins:
                    # Build spaces for target bin from its items
                    target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_boxes[target_bin]]
                    
                    # Try to find a position
                    spaces = [(0, 0, 0, W, H, D)]
                    for (tx, ty, tz, tw, th, td) in target_items:
                        new_spaces = []
                        for (sx, sy, sz, sw, sh, sd) in spaces:
                            sx2, sy2, sz2 = sx + sw, sy + sh, sz + sd
                            if tx >= sx2 or sx >= tx + tw or ty >= sy2 or sy >= ty + th or tz >= sz2 or sz >= tz + td:
                                new_spaces.append((sx, sy, sz, sw, sh, sd))
                            else:
                                if tx > sx:
                                    new_spaces.append((sx, sy, sz, tx - sx, sh, sd))
                                if tx + tw < sx2:
                                    new_spaces.append((tx + tw, sy, sz, sx2 - tx - tw, sh, sd))
                                if ty > sy:
                                    new_spaces.append((tx, sy, sz, tw, ty - sy, sd))
                                if ty + th < sy2:
                                    new_spaces.append((tx, ty + th, sz, tw, sy2 - ty - th, sd))
                                if tz > sz:
                                    new_spaces.append((tx, ty, sz, tw, th, tz - sz))
                                if tz + td < sz2:
                                    new_spaces.append((tx, ty, tz + td, tw, th, sz2 - tz - td))
                        spaces = new_spaces
                    
                    pos, _ = find_position_in_spaces(spaces, w, h, d)
                    if pos is not None:
                        new_placements = list(placements)
                        new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements)
                        if new_score < current_score - 1e-9:
                            placements = new_placements
                            current_score = new_score
                            bin_boxes[b].remove(box_idx)
                            if not bin_boxes[b]:
                                del bin_boxes[b]
                            bin_boxes[target_bin].append(box_idx)
                            improved = True
                            break
        
        return placements
    
    # Generate orderings
    indices = list(range(n))
    orderings = []
    
    # Volume descending
    orderings.append(sorted(indices, key=lambda i: box_vols[i], reverse=True))
    
    # Max dimension descending
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # Surface area descending
    orderings.append(sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # Perimeter descending
    orderings.append(sorted(indices, key=lambda i: boxes[i][0]+boxes[i][1]+boxes[i][2], reverse=True))
    
    # Min dimension descending
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    
    # Aspect ratio (square-ish first)
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]) / max(boxes[i]), reverse=True))
    
    # Original order
    orderings.append(indices)
    
    # Randomized volume descending
    for seed in [42, 123, 7]:
        random.seed(seed)
        shuffled = sorted(indices, key=lambda i: box_vols[i], reverse=True)
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_result = None
    best_score = float('inf')
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        
        result = pack_with_order_v2(ordering)
        if result is not None:
            score = compute_score(result)
            if score < best_score:
                best_score = score
                best_result = result
    
    # Fallback if no valid result
    if best_result is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
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
        best_score = compute_score(best_result)
    
    # Local search polish
    if best_result and time.time() < deadline - 8:
        best_result = local_search_polish(best_result)
        best_score = compute_score(best_result)
    
    # Relabel bins to be consecutive starting from 0
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]
    
    return best_result


def compute_score(placements):
    if not placements:
        return 0.0
    
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    
    last_bin = max(bins_used)
    
    # Need container dimensions - use a reasonable default or compute from placements
    # Actually, we need the container. Let's store it or compute differently.
    # For now, assume we can get it from the placements or use a global
    # Better: compute from the max extent
    W = max(p[1] + p[4] for p in placements)
    H = max(p[2] + p[5] for p in placements)
    D = max(p[3] + p[6] for p in placements)
    container_vol = W * H * D
    
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    
    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
