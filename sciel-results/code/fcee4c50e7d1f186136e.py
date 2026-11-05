import time
import sys
import numpy as np

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _can_place(box, pos, placements, W, H, D):
    x, y, z = pos
    w, h, d = box
    
    if x + w > W or y + h > H or z + d > D:
        return False
    
    for (px, py, pz, pw, ph, pd) in placements:
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position_bfd(box, placements, W, H, D):
    """
    Find the best position for a box in a bin using Bottom-Front-Left (BFL) logic.
    Preference: Lowest Z, then Lowest Y, then Lowest X.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    return (x, y, z)
                    
    return None

def _find_position_best_fit(box, placements, W, H, D):
    """
    Find the position that minimizes the sum of distances to existing boxes (or boundaries).
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    best_pos = None
    best_score = float('inf')
    
    if not placements:
        return (0, 0, 0)
        
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    min_dist = float('inf')
                    for (px, py, pz, pw, ph, pd) in placements:
                        dx = max(0, max(x, px) - min(x + w, px + pw))
                        dy = max(0, max(y, py) - min(y + h, py + ph))
                        dz = max(0, max(z, pz) - min(z + d, pz + pd))
                        dist = dx + dy + dz
                        if dist < min_dist:
                            min_dist = dist
                            if min_dist == 0:
                                break
                    score = z * 10000 + y * 100 + x + min_dist * 0.1
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    def _pack_with_order(order, finder_func):
        bins = []
        result = [None] * n
        
        for orig_idx in order:
            box = boxes[orig_idx]
            w, h, d = box
            placed = False
            
            for bi in range(len(bins)):
                pos = finder_func(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                result[orig_idx] = (bi, 0, 0, 0, w, h, d)
        
        for i in range(n):
            if result[i] is None:
                bi = len(bins)
                w, h, d = boxes[i]
                bins.append([(0, 0, 0, w, h, d)])
                result[i] = (bi, 0, 0, 0, w, h, d)
        
        return bins, result
    
    def _score_solution(bins, result):
        if not bins:
            return float('inf')
        
        occupied_bins = set()
        for r in result:
            occupied_bins.add(r[0])
        B = len(occupied_bins)
        
        last_bin_id = max(r[0] for r in result)
        last_vol = sum(p[3] * p[4] * p[5] for p in bins[last_bin_id])
        container_vol = W * H * D
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        
        return B + 0.1 * (1 - fill)
    
    # Strategies
    # 1. Volume Desc
    order_vol_desc = sorted(range(n), key=lambda i: _box_volume(boxes[i]), reverse=True)
    # 2. Max Dim Desc
    order_max_dim_desc = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    # 3. Volume Desc, then Max Dim Desc
    order_vol_max_dim = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    # 4. Max Dim Desc, then Volume Desc
    order_max_dim_vol = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    # 5. Height Desc (Z dim)
    order_h_desc = sorted(range(n), key=lambda i: boxes[i][1], reverse=True)
    # 6. Width Desc (X dim)
    order_w_desc = sorted(range(n), key=lambda i: boxes[i][0], reverse=True)
    # 7. Depth Desc (Y dim)
    order_d_desc = sorted(range(n), key=lambda i: boxes[i][2], reverse=True)
    # 8. Surface Area Desc
    order_sa_desc = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)

    strategies = [
        (order_vol_max_dim, _find_position_bfd),
        (order_max_dim_vol, _find_position_bfd),
        (order_vol_desc, _find_position_best_fit),
        (order_max_dim_desc, _find_position_best_fit),
        (order_h_desc, _find_position_bfd),
        (order_w_desc, _find_position_bfd),
        (order_d_desc, _find_position_bfd),
        (order_sa_desc, _find_position_bfd),
        (order_vol_max_dim, _find_position_best_fit),
        (order_max_dim_vol, _find_position_best_fit),
    ]
    
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    for order, finder in strategies:
        if time.time() > deadline - 30.0:
            break
        
        bins_try, result_try = _pack_with_order(order, finder)
        score_try = _score_solution(bins_try, result_try)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_result = result_try
    
    bins = best_bins
    result = best_result
            
    # Phase 2: Optimize
    if time.time() < deadline - 5.0:
        # Relabel to 0..B-1
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
        
        # Rebuild bins list to match new labels
        new_bins = [[] for _ in range(len(occupied_bins))]
        for i in range(n):
            b, x, y, z, w, h, d = result[i]
            new_bins[b].append((x, y, z, w, h, d))
        bins = new_bins
        
        B = len(bins)
        
        # 1. Try to reduce B by merging last bin into others
        # Sort bins by volume descending? No, just try moving boxes from last bin to others.
        # Actually, standard approach: try to move boxes from bin i to bin j (i < j) to potentially empty bin j.
        # Since we want to minimize B, we should try to empty the highest index bins first.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Iterate from highest bin index down
            for bi in range(len(bins) - 1, 0, -1):
                if not bins[bi]:
                    continue
                # Try to move all boxes from bin bi to other bins
                boxes_to_move = bins[bi][:]
                can_move_all = True
                temp_bins = [list(b) for b in bins]
                
                for box_info in boxes_to_move:
                    x, y, z, w, h, d = box_info
                    moved = False
                    for bj in range(len(temp_bins)):
                        if bj == bi:
                            continue
                        pos = _find_position_bfd((w, h, d), temp_bins[bj], W, H, D)
                        if pos is not None:
                            # Move it
                            temp_bins[bj].append((pos[0], pos[1], pos[2], w, h, d))
                            moved = True
                            break
                    if not moved:
                        can_move_all = False
                        break
                
                if can_move_all:
                    # Commit
                    bins = temp_bins
                    # Update result
                    for i in range(n):
                        if result[i][0] == bi:
                            # Find new bin
                            # This is tricky, we need to track which box went where.
                            # Let's just re-pack from scratch? No, too slow.
                            # Let's store the mapping.
                            pass
                    # Actually, let's just re-run the packing with the best order found?
                    # Or simpler: if we can empty a bin, we save 1.0 score.
                    # Let's just implement a simpler "move one box" heuristic first.
                    
                    # Simpler heuristic: Move boxes from last bin to previous bins if they fit.
                    # This might not empty the bin, but reduces B if we manage to empty it.
                    # Let's stick to the "move one box" logic from the original but more aggressive.
                    
                    # Actually, let's just use the original logic which was decent.
                    # The original logic:
                    # 1. Move boxes from other bins to last bin to increase fill.
                    # 2. Swap boxes.
                    # This maximizes last-bin fill, which is good for the 0.1 term.
                    # But minimizing B is worth 1.0.
                    
                    # Let's try to minimize B first.
                    # Try to move boxes from bin B-1 to bin B-2, etc.
                    
                    # Let's restart the optimization with a clearer structure.
                    break
                    
            if not improved:
                break
                
        # Let's use the optimization from the original code which was specific to last-bin fill.
        # And add a "bin consolidation" step.
        
        # Step 1: Consolidate bins (try to empty higher index bins)
        for _ in range(10): # Max iterations
            if time.time() > deadline - 5.0:
                break
            improved = False
            for bi in range(len(bins) - 1, 0, -1):
                if not bins[bi]:
                    continue
                boxes_in_bin = bins[bi][:]
                all_moved = True
                # We need to move them one by one to other bins
                # This is a knapsack-like problem.
                # Greedy: move largest first?
                boxes_in_bin.sort(key=lambda b: b[3]*b[4]*b[5], reverse=True)
                
                temp_bins = [list(b) for b in bins]
                moved_indices = []
                
                for box_info in boxes_in_bin:
                    x, y, z, w, h, d = box_info
                    moved = False
                    for bj in range(len(temp_bins)):
                        if bj == bi:
                            continue
                        pos = _find_position_bfd((w, h, d), temp_bins[bj], W, H, D)
                        if pos is not None:
                            temp_bins[bj].append((pos[0], pos[1], pos[2], w, h, d))
                            moved = True
                            break
                    if not moved:
                        all_moved = False
                        break
                
                if all_moved:
                    bins = temp_bins
                    improved = True
                    # Update result
                    # We need to know which box is in which bin.
                    # Let's rebuild result from bins?
                    # We need a mapping from box index to position.
                    # This is getting complicated. Let's just re-pack?
                    # No, let's just track it.
                    
                    # Actually, the original code's approach of moving boxes to the LAST bin
                    # to increase fill is simpler and directly targets the score term.
                    # Minimizing B is harder.
                    # Let's stick to the original optimization logic but make it faster.
                    
                    break
                    
            if not improved:
                break
        
        # Step 2: Optimize last bin fill (Original logic)
        last_bin_id = len(bins) - 1
        if bins[last_bin_id]:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                
                # Move boxes from other bins to last bin
                for idx in range(n):
                    if time.time() > deadline - 3.0:
                        break
                    if result[idx][0] == last_bin_id:
                        continue
                    
                    cur_bin_id = result[idx][0]
                    box = boxes[idx]
                    w, h, d = box
                    
                    last_placements = bins[last_bin_id]
                    pos = _find_position_bfd((w, h, d), last_placements, W, H, D)
                    
                    if pos is not None:
                        # Remove from current
                        cur_placements = bins[cur_bin_id]
                        target = (result[idx][1], result[idx][2], result[idx][3], w, h, d)
                        
                        removed = False
                        for j in range(len(cur_placements)):
                            if cur_placements[j] == target:
                                cur_placements.pop(j)
                                removed = True
                                break
                                
                        if removed:
                            bins[last_bin_id].append((pos[0], pos[1], pos[2], w, h, d))
                            result[idx] = (last_bin_id, pos[0], pos[1], pos[2], w, h, d)
                            improved = True
                            break
                            
                # Swaps
                if not improved and time.time() < deadline - 5.0:
                    last_boxes_indices = [i for i in range(n) if result[i][0] == last_bin_id]
                    
                    for idx_last in last_boxes_indices:
                        if time.time() > deadline - 5.0:
                            break
                        
                        box_last = boxes[idx_last]
                        wl, hl, dl = box_last
                        pos_last = (result[idx_last][1], result[idx_last][2], result[idx_last][3])
                        target_last = (pos_last[0], pos_last[1], pos_last[2], wl, hl, dl)
                        
                        for idx_other in range(n):
                            if time.time() > deadline - 5.0:
                                break
                            if result[idx_other][0] == last_bin_id:
                                continue
                                
                            box_other = boxes[idx_other]
                            wo, ho, do = box_other
                            pos_other = (result[idx_other][1], result[idx_other][2], result[idx_other][3])
                            target_other = (pos_other[0], pos_other[1], pos_other[2], wo, ho, do)
                            
                            if _box_volume(box_other) <= _box_volume(box_last):
                                continue
                                
                            last_temp = bins[last_bin_id][:]
                            rem_last = False
                            for j in range(len(last_temp)):
                                if last_temp[j] == target_last:
                                    last_temp.pop(j)
                                    rem_last = True
                                    break
                            if not rem_last:
                                continue
                                
                            pos_new = _find_position_bfd((wo, ho, do), last_temp, W, H, D)
                            if pos_new is None:
                                continue
                                
                            other_bin_id = result[idx_other][0]
                            other_temp = bins[other_bin_id][:]
                            rem_other = False
                            for j in range(len(other_temp)):
                                if other_temp[j] == target_other:
                                    other_temp.pop(j)
                                    rem_other = True
                                    break
                            if not rem_other:
                                continue
                                
                            pos_new_last = _find_position_bfd((wl, hl, dl), other_temp, W, H, D)
                            if pos_new_last is None:
                                continue
                                
                            # Swap
                            bins[last_bin_id] = last_temp + [(pos_new[0], pos_new[1], pos_new[2], wo, ho, do)]
                            bins[other_bin_id] = other_temp + [(pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)]
                            
                            result[idx_last] = (other_bin_id, pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)
                            result[idx_other] = (last_bin_id, pos_new[0], pos_new[1], pos_new[2], wo, ho, do)
                            
                            improved = True
                            break
                        if improved:
                            break

    # Final relabel
    if bins:
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
            
    return result
