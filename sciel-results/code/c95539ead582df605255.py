<<<<<<< SEARCH
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
=======
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

def _calculate_score(result, boxes, container):
    """Calculate the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    
    # Find occupied bins and their volumes
    bin_vols = {}
    for r in result:
        bin_id = r[0]
        if bin_id not in bin_vols:
            bin_vols[bin_id] = 0
        w, h, d = r[4], r[5], r[6]
        bin_vols[bin_id] += w * h * d
    
    if not bin_vols:
        return 0.0
    
    B = len(bin_vols)
    last_bin_id = max(bin_vols.keys())
    last_bin_vol = bin_vols[last_bin_id]
    last_bin_fraction = last_bin_vol / container_vol if container_vol > 0 else 1.0
    
    return B + 0.1 * (1 - last_bin_fraction)

def _sa_optimize(result, bins, bin_grids, boxes, container, deadline):
    """Simulated Annealing local search to optimize the packing."""
    W, H, D = container
    n = len(boxes)
    random.seed(42)
    
    # Current state
    current_result = [r[:] for r in result]
    current_score = _calculate_score(current_result, boxes, container)
    
    # Best state
    best_result = [r[:] for r in result]
    best_score = current_score
    
    # SA parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    max_iter = 5000
    
    def _rebuild_state(result_state):
        """Rebuild bins and grids from a result state."""
        new_bins = {}
        new_grids = {}
        for r in result_state:
            bin_id = r[0]
            if bin_id not in new_bins:
                new_bins[bin_id] = []
                new_grids[bin_id] = np.zeros((W, H, D), dtype=bool)
            x, y, z, w, h, d = r[1], r[2], r[3], r[4], r[5], r[6]
            new_bins[bin_id].append((bin_id, x, y, z, w, h, d))
            new_grids[bin_id][x:x+w, y:y+h, z:z+d] = True
        return new_bins, new_grids
    
    def _try_move(box_idx, new_bin_id):
        """Try to move a box to a new bin. Returns True if successful."""
        old_r = current_result[box_idx]
        old_bin = old_r[0]
        box = boxes[box_idx]
        w, h, d = box
        old_pos = (old_r[1], old_r[2], old_r[3])
        
        # Remove from old bin
        if old_bin in new_grids:
            new_grids[old_bin][old_pos[0]:old_pos[0]+w, old_pos[1]:old_pos[1]+h, old_pos[2]:old_pos[2]+d] = False
            new_bins[old_bin] = [p for p in new_bins[old_bin] if not (p[1] == old_pos[0] and p[2] == old_pos[1] and p[3] == old_pos[2] and p[4] == w and p[5] == h and p[6] == d)]
        
        # Try to place in new bin
        if new_bin_id not in new_grids:
            new_grids[new_bin_id] = np.zeros((W, H, D), dtype=bool)
            new_bins[new_bin_id] = []
        
        pos = _find_position_grid(box, new_bins[new_bin_id], new_grids[new_bin_id], W, H, D)
        if pos is None:
            # Restore
            new_grids[old_bin][old_pos[0]:old_pos[0]+w, old_pos[1]:old_pos[1]+h, old_pos[2]:old_pos[2]+d] = True
            new_bins[old_bin].append((old_bin, old_pos[0], old_pos[1], old_pos[2], w, h, d))
            return False
        
        # Place in new bin
        x, y, z = pos
        new_grids[new_bin_id][x:x+w, y:y+h, z:z+d] = True
        new_bins[new_bin_id].append((new_bin_id, x, y, z, w, h, d))
        current_result[box_idx] = (new_bin_id, x, y, z, w, h, d)
        return True
    
    # Main SA loop
    iter_count = 0
    while T > T_min and time.time() < deadline - 2.0 and iter_count < max_iter:
        iter_count += 1
        if iter_count % 100 == 0 and time.time() > deadline - 2.0:
            break
        
        # Choose a random box
        box_idx = random.randint(0, n - 1)
        
        # Choose a target bin (random existing bin or new bin)
        occupied_bin_ids = list(set(r[0] for r in current_result))
        if random.random() < 0.7 and occupied_bin_ids:
            new_bin_id = random.choice(occupied_bin_ids)
        else:
            new_bin_id = max(occupied_bin_ids) + 1 if occupied_bin_ids else 0
        
        # Save old state
        old_bin = current_result[box_idx][0]
        old_pos = (current_result[box_idx][1], current_result[box_idx][2], current_result[box_idx][3])
        
        # Try the move
        if _try_move(box_idx, new_bin_id):
            new_score = _calculate_score(current_result, boxes, container)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_result = [r[:] for r in current_result]
            else:
                # Reject: restore
                _try_move(box_idx, old_bin)
        
        # Cool down
        T *= cooling_rate
    
    # Apply best result
    for i in range(n):
        result[i] = best_result[i]
    
    # Rebuild bins and grids from best result
    new_bins, new_grids = _rebuild_state(best_result)
    bins.clear()
    bin_grids.clear()
    for bin_id in sorted(new_bins.keys()):
        bins.append(new_bins[bin_id])
        bin_grids.append(new_grids[bin_id])
>>>>>>> REPLACE
