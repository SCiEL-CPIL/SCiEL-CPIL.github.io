"""
3D Bin Packing Solution
Strategy:
1. Generate multiple packing solutions using different sorting orders and placement heuristics.
   - Orders: Volume Desc, Max Dim Desc, Surface Area Desc.
   - Heuristics: First-Fit (Bottom-Fill-Left), Best-Fit (Min distance to walls/boxes).
2. Select the solution with the best score (B + 0.1 * (1 - last_bin_fill)).
3. Post-processing:
   - Move boxes into the last bin to maximize its fill ratio.
   - Try to consolidate bins to reduce B (number of occupied bins).
4. Relabel bins to be consecutive.
"""
import time
import numpy as np

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _box_surface(box):
    w, h, d = box
    return 2 * (w*h + h*d + w*d)

def _update_grid(grid, box, pos, add=True):
    x, y, z = pos
    w, h, d = box
    grid[x:x+w, y:y+h, z:z+d] = add

def _find_position_bfl(box, placements, grid, W, H, D):
    """
    Find position using Bottom-Fill-Left (First Fit).
    Returns the first valid position in sorted candidate order.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Candidate coordinates: 0 and right/top/front faces of existing boxes
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    # BFL: iterate z (lowest first), then y (lowest first), then x (lowest first)
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def _find_position_best_fit(box, placements, grid, W, H, D):
    """
    Find position using Best-Fit heuristic with max-contact scoring.
    Prefers positions with maximum contact area with existing boxes and walls.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    if not placements:
        return (0, 0, 0)
        
    best_pos = None
    best_score = float('inf')
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if grid[x:x+w, y:y+h, z:z+d].any():
                    continue
                
                # Contact with walls
                contact = 0
                if x == 0:
                    contact += h * d
                if x + w == W:
                    contact += h * d
                if y == 0:
                    contact += w * d
                if y + h == H:
                    contact += w * d
                if z == 0:
                    contact += w * h
                if z + d == D:
                    contact += w * h
                
                # Contact with existing boxes
                for (_, px, py, pz, pw, ph, pd) in placements:
                    # Check each of the 6 faces
                    # Box's +x face at x+w touches box's -x face at px
                    if x + w == px:
                        oy = max(y, py)
                        oy2 = min(y + h, py + ph)
                        oz = max(z, pz)
                        oz2 = min(z + d, pz + pd)
                        if oy < oy2 and oz < oz2:
                            contact += (oy2 - oy) * (oz2 - oz)
                    # Box's -x face at x touches box's +x face at px+pw
                    if x == px + pw:
                        oy = max(y, py)
                        oy2 = min(y + h, py + ph)
                        oz = max(z, pz)
                        oz2 = min(z + d, pz + pd)
                        if oy < oy2 and oz < oz2:
                            contact += (oy2 - oy) * (oz2 - oz)
                    # Box's +y face
                    if y + h == py:
                        ox = max(x, px)
                        ox2 = min(x + w, px + pw)
                        oz = max(z, pz)
                        oz2 = min(z + d, pz + pd)
                        if ox < ox2 and oz < oz2:
                            contact += (ox2 - ox) * (oz2 - oz)
                    # Box's -y face
                    if y == py + ph:
                        ox = max(x, px)
                        ox2 = min(x + w, px + pw)
                        oz = max(z, pz)
                        oz2 = min(z + d, pz + pd)
                        if ox < ox2 and oz < oz2:
                            contact += (ox2 - ox) * (oz2 - oz)
                    # Box's +z face
                    if z + d == pz:
                        ox = max(x, px)
                        ox2 = min(x + w, px + pw)
                        oy = max(y, py)
                        oy2 = min(y + h, py + ph)
                        if ox < ox2 and oy < oy2:
                            contact += (ox2 - ox) * (oy2 - oy)
                    # Box's -z face
                    if z == pz + pd:
                        ox = max(x, px)
                        ox2 = min(x + w, px + pw)
                        oy = max(y, py)
                        oy2 = min(y + h, py + ph)
                        if ox < ox2 and oy < oy2:
                            contact += (ox2 - ox) * (oy2 - oy)
                
                # Score: maximize contact (so minimize negative contact)
                # Tie-break with BFL preference
                score = -contact + z * 0.001 + y * 0.0001 + x * 0.00001
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
                        
    return best_pos

def _pack_strategy(boxes, container, order, finder_func, deadline):
    """
    Execute a specific packing strategy.
    Returns (bins, bin_grids, result) or None if timeout.
    """
    W, H, D = container
    n = len(boxes)
    
    bins = []
    bin_grids = []
    result = [None] * n
    
    for orig_idx in order:
        if time.time() > deadline - 5.0:
            return None
            
        box = boxes[orig_idx]
        w, h, d = box
        placed = False
        
        if bins:
            # Try to place in existing bins
            # Sort bins by remaining volume ascending (tightest first) for BFD-like behavior
            bin_indices = list(range(len(bins)))
            bin_vols = []
            for bi in bin_indices:
                vol = np.sum(bin_grids[bi])
                bin_vols.append((W*H*D - vol, bi))
            bin_vols.sort(key=lambda x: x[0])
            
            for _, bi in bin_vols:
                pos = finder_func(box, bins[bi], bin_grids[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    _update_grid(bin_grids[bi], box, pos)
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, box, (0, 0, 0))
            bin_grids.append(grid)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
            
    # Safety check: ensure all placed
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0, 0, 0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    return bins, bin_grids, result

def _score_solution(bins, result, W, H, D):
    if not bins or not result:
        return float('inf')
        
    occupied_bins = set()
    for r in result:
        occupied_bins.add(r[0])
    B = len(occupied_bins)
    
    last_bin_id = max(r[0] for r in result)
    
    # Calculate fill of last bin
    last_vol = 0
    for p in bins[last_bin_id]:
        _, x, y, z, w, h, d = p
        last_vol += w * h * d
        
    container_vol = W * H * D
    fill = last_vol / container_vol if container_vol > 0 else 0.0
    
    return B + 0.1 * (1 - fill)

def _optimize_last_bin(bins, bin_grids, result, boxes, W, H, D, deadline):
    """
    Move boxes from other bins into the last bin to increase its fill ratio.
    """
    if not result:
        return
        
    last_bin_id = max(r[0] for r in result)
    
    moved_any = True
    while moved_any and time.time() < deadline - 2.0:
        moved_any = False
        # Sort candidates by box volume descending to fill with larger boxes first
        candidates = sorted(
            [i for i in range(len(result)) if result[i][0] != last_bin_id],
            key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2],
            reverse=True
        )
        
        for idx in candidates:
            if time.time() > deadline - 2.0:
                break
                
            cur_bin = result[idx][0]
            box = boxes[idx]
            w, h, d = box
            box_pos = result[idx][1:4]
            
            # Temporarily remove from source bin
            _update_grid(bin_grids[cur_bin], box, box_pos, add=False)
            
            # Try to place in last bin using Best Fit to maximize tightness
            pos = _find_position_best_fit(box, bins[last_bin_id], bin_grids[last_bin_id], W, H, D)
            
            if pos is not None:
                # Move it
                x, y, z = pos
                bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                _update_grid(bin_grids[last_bin_id], box, pos)
                
                # Remove from source bin's placement list
                bins[cur_bin] = [p for p in bins[cur_bin] 
                                 if not (p[1] == box_pos[0] and p[2] == box_pos[1] 
                                         and p[3] == box_pos[2] and p[4] == w 
                                         and p[5] == h and p[6] == d)]
                result[idx] = (last_bin_id, x, y, z, w, h, d)
                moved_any = True
                break
            else:
                # Restore if it didn't fit
                _update_grid(bin_grids[cur_bin], box, box_pos, add=True)

def _consolidate_bins(bins, bin_grids, result, boxes, W, H, D, deadline):
    """
    Try to empty non-last bins by moving their boxes to other bins.
    Uses a greedy per-box approach with rollback on failure.
    """
    if not result:
        return
        
    last_bin_id = max(r[0] for r in result)
    
    # Try to empty non-last bins (highest index first, excluding last_bin_id)
    non_last_bins = sorted([bi for bi in range(len(bins)) if bi != last_bin_id and bins[bi]], reverse=True)
    
    for src_bin in non_last_bins:
        if time.time() > deadline - 3.0:
            break
            
        boxes_to_move = list(bins[src_bin])
        if not boxes_to_move:
            continue
        
        # Build index map for this bin
        bin_box_indices = []
        for i in range(len(result)):
            if result[i] and result[i][0] == src_bin:
                bin_box_indices.append(i)
        
        if not bin_box_indices:
            bins[src_bin] = []
            continue
        
        can_move_all = True
        saved_states = []
        
        # Sort boxes by volume descending to move larger ones first
        bin_box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for box_idx in bin_box_indices:
            if time.time() > deadline - 3.0:
                can_move_all = False
                break
                
            p = result[box_idx]
            x, y, z, w, h, d = p[1:7]
            box = (w, h, d)
            
            # Remove from source
            _update_grid(bin_grids[src_bin], box, (x, y, z), add=False)
            saved_states.append((box_idx, x, y, z, w, h, d))
            
            # Try to place in other bins (not src_bin)
            placed = False
            for dst_bin in range(len(bins)):
                if dst_bin == src_bin:
                    continue
                pos = _find_position_bfl(box, bins[dst_bin], bin_grids[dst_bin], W, H, D)
                if pos is not None:
                    nx, ny, nz = pos
                    bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                    _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                    result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                    placed = True
                    break
                    
            if not placed:
                # Restore all moved boxes
                for bi2, xi, yi, zi, wi, hi, di in saved_states:
                    _update_grid(bin_grids[bi2 if False else src_bin], (wi, hi, di), (xi, yi, zi), add=True)
                can_move_all = False
                break
                
        if can_move_all:
            bins[src_bin] = []
            bin_grids[src_bin] = np.zeros((W, H, D), dtype=bool)

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate multiple ordering strategies
    strategies = []
    
    # Strategy 1: Volume descending, BFL
    order1 = sorted(range(n), key=lambda i: _box_volume(boxes[i]), reverse=True)
    strategies.append((order1, _find_position_bfl))
    
    # Strategy 2: Max dimension descending, BFL
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append((order2, _find_position_bfl))
    
    # Strategy 3: Surface area descending, BFL
    order3 = sorted(range(n), key=lambda i: _box_surface(boxes[i]), reverse=True)
    strategies.append((order3, _find_position_bfl))
    
    # Strategy 4: Volume descending, Best Fit
    strategies.append((order1, _find_position_best_fit))
    
    # Strategy 5: Max dimension descending, Best Fit
    strategies.append((order2, _find_position_best_fit))
    
    # Strategy 6: Surface area descending, Best Fit
    strategies.append((order3, _find_position_best_fit))
    
    # Try all strategies and pick the best
    best_score = float('inf')
    best_bins = None
    best_grids = None
    best_result = None
    
    for order, finder in strategies:
        if time.time() > deadline - 20.0:
            break
            
        outcome = _pack_strategy(boxes, container, order, finder, deadline)
        if outcome is None:
            continue
            
        bins_try, grids_try, result_try = outcome
        score_try = _score_solution(bins_try, result_try, W, H, D)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_grids = grids_try
            best_result = result_try
            
    if best_bins is None:
        # Fallback: emergency placement
        bins = []
        bin_grids = []
        result = [None] * n
        for i in range(n):
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0, 0, 0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
        return result
        
    bins = best_bins
    bin_grids = best_grids
    result = best_result
    
    # Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 10.0:
        _optimize_last_bin(bins, bin_grids, result, boxes, W, H, D, deadline)
        
    # Phase 3: Consolidate Bins
    if time.time() < deadline - 5.0:
        _consolidate_bins(bins, bin_grids, result, boxes, W, H, D, deadline)
        
    # Phase 4: Second pass of last-bin optimization after consolidation
    if time.time() < deadline - 2.0:
        _optimize_last_bin(bins, bin_grids, result, boxes, W, H, D, deadline)
        
    # 4. Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
            
    return result
