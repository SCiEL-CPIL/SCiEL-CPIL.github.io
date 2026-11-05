import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(0, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    def check_fit(x, y, z, w, h, d, bin_boxes):
        """Check if box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True
    
    def get_candidates_fast(box, bin_boxes):
        """Get candidate positions for a box in a bin. Returns list of (x,y,z)."""
        w, h, d = box
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidates from existing boxes' faces
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Right face
            candidates.add((bx + bw, by, bz))
            # Top face (in y)
            candidates.add((bx, by + bh, bz))
            # Back face (in z)
            candidates.add((bx, by, bz + bd))
            
            # Also try combinations of faces for better packing
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
        
        # Filter valid candidates
        valid = []
        for cx, cy, cz in candidates:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                if check_fit(cx, cy, cz, w, h, d, bin_boxes):
                    valid.append((cx, cy, cz))
        
        return valid
    
    def score_position_bfl(pos, box, bin_boxes):
        """Bottom-Fill-Left: minimize z, then y, then x."""
        cx, cy, cz = pos
        return (cz, cy, cx)
    
    def score_position_best_fit(pos, box, bin_boxes, W, H, D):
        """Best fit: minimize sum of distances to nearest surfaces/boxes."""
        cx, cy, cz = pos
        w, h, d = box
        
        # Distance to walls
        dist_x = min(cx, W - (cx + w))
        dist_y = min(cy, H - (cy + h))
        dist_z = min(cz, D - (cz + d))
        
        # Distance to existing boxes
        min_dist_to_box = float('inf')
        for bx, by, bz, bw, bh, bd in bin_boxes:
            dx = 0
            if cx + w <= bx:
                dx = bx - (cx + w)
            elif cx >= bx + bw:
                dx = cx - (bx + bw)
            
            dy = 0
            if cy + h <= by:
                dy = by - (cy + h)
            elif cy >= by + bh:
                dy = cy - (by + bh)
            
            dz = 0
            if cz + d <= bz:
                dz = bz - (cz + d)
            elif cz >= bz + bd:
                dz = cz - (bz + bd)
            
            dist = max(dx, dy, dz)
            if dist < min_dist_to_box:
                min_dist_to_box = dist
            if min_dist_to_box == 0:
                break
        
        # Primary: best fit (tightness)
        # Secondary: bottom-fill
        # Tertiary: back-fill
        # Quaternary: left-fill
        return (dist_x + dist_y + dist_z + min_dist_to_box, cz, cy, cx)
    
    def run_strategy(sort_key, position_scorer):
        """Run a single packing strategy. Returns (score, placements)."""
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=sort_key, reverse=True)
        
        bins = []  # List of lists of (x, y, z, w, h, d)
        placements = [None] * n
        
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            
            # Try to place in existing bins
            best_bin_idx = -1
            best_pos = None
            best_score = None
            
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates_fast(box, bin_boxes)
                if not candidates:
                    continue
                
                # Find best position in this bin
                for pos in candidates:
                    s = position_scorer(pos, box, bin_boxes, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = pos
                        best_bin_idx = b_idx
            
            if best_bin_idx != -1:
                cx, cy, cz = best_pos
                bins[best_bin_idx].append((cx, cy, cz, w, h, d))
                placements[orig_idx] = (best_bin_idx, cx, cy, cz, w, h, d)
            else:
                # Open new bin
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        
        if not bins:
            return float('inf'), placements
        
        B = len(bins)
        last_bin = bins[-1]
        occupied_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        fill = occupied_vol / container_vol
        strategy_score = B + 0.1 * (1 - fill)
        
        return strategy_score, placements
    
    # Define sorting strategies
    def sort_volume_desc(x):
        return x[1][0]*x[1][1]*x[1][2]
    
    def sort_maxdim_desc(x):
        return max(x[1])
    
    def sort_volume_then_maxdim(x):
        return (x[1][0]*x[1][1]*x[1][2], max(x[1]))
    
    def sort_height_desc(x):
        return x[1][2]
    
    def sort_width_desc(x):
        return x[1][0]
    
    def sort_depth_desc(x):
        return x[1][1]
    
    def sort_surface_area_desc(x):
        w, h, d = x[1]
        return 2*(w*h + h*d + w*d)
    
    # Define position scoring strategies
    def make_bfl_scorer():
        return lambda pos, box, bin_boxes, W, H, D: score_position_bfl(pos, box, bin_boxes)
    
    def make_best_fit_scorer():
        return lambda pos, box, bin_boxes, W, H, D: score_position_best_fit(pos, box, bin_boxes, W, H, D)
    
    bfl_scorer = make_bfl_scorer()
    best_fit_scorer = make_best_fit_scorer()
    
    strategies = [
        (sort_volume_then_maxdim, bfl_scorer),
        (sort_volume_desc, bfl_scorer),
        (sort_maxdim_desc, bfl_scorer),
        (sort_height_desc, bfl_scorer),
        (sort_width_desc, bfl_scorer),
        (sort_depth_desc, bfl_scorer),
        (sort_surface_area_desc, bfl_scorer),
        (sort_volume_then_maxdim, best_fit_scorer),
        (sort_volume_desc, best_fit_scorer),
        (sort_maxdim_desc, best_fit_scorer),
        (sort_surface_area_desc, best_fit_scorer),
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for sort_key, scorer in strategies:
        if time.time() > deadline - 10.0:
            break
        
        s, placements = run_strategy(sort_key, scorer)
        if s < best_score:
            best_score = s
            best_placements = placements
    
    # If no strategy completed, use a simple fallback
    if best_placements is None:
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=sort_volume_desc, reverse=True)
        bins = []
        placements = [None] * n
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates_fast(box, bin_boxes)
                if candidates:
                    # Pick the position with minimum (z, y, x)
                    best_pos = min(candidates, key=lambda p: (p[2], p[1], p[0]))
                    cx, cy, cz = best_pos
                    bins[b_idx].append((cx, cy, cz, w, h, d))
                    placements[orig_idx] = (b_idx, cx, cy, cz, w, h, d)
                    placed = True
                    break
            if not placed:
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        best_placements = placements
    
    # Phase 2: Consolidation - try to move boxes to reduce bin count
    if time.time() < deadline - 5.0:
        # Extract bin assignments from placements
        bin_assignments = {}
        for i, p in enumerate(best_placements):
            bin_id = p[0]
            if bin_id not in bin_assignments:
                bin_assignments[bin_id] = []
            bin_assignments[bin_id].append(i)
        
        # Get unique bin ids
        bin_ids = sorted(bin_assignments.keys())
        
        # Try to move boxes from higher-numbered bins to lower-numbered bins
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            for src_bin in reversed(bin_ids):
                if time.time() > deadline - 3.0:
                    break
                
                if src_bin not in bin_assignments:
                    continue
                
                boxes_in_bin = list(bin_assignments[src_bin])
                
                for box_idx in boxes_in_bin:
                    if time.time() > deadline - 3.0:
                        break
                    
                    box = boxes[box_idx]
                    w, h, d = box
                    placed_elsewhere = False
                    
                    for dst_bin in bin_ids:
                        if dst_bin == src_bin:
                            continue
                        if dst_bin not in bin_assignments:
                            continue
                        
                        # Get boxes in destination bin
                        dst_boxes = []
                        for other_idx in bin_assignments[dst_bin]:
                            if other_idx == box_idx:
                                continue
                            op = best_placements[other_idx]
                            dst_boxes.append((op[1], op[2], op[3], op[4], op[5], op[6]))
                        
                        candidates = get_candidates_fast(box, dst_boxes)
                        if candidates:
                            # Find best position
                            best_pos = min(candidates, key=lambda p: (p[2], p[1], p[0]))
                            cx, cy, cz = best_pos
                            
                            # Update placement
                            best_placements[box_idx] = (dst_bin, cx, cy, cz, w, h, d)
                            
                            # Update bin assignments
                            bin_assignments[dst_bin].append(box_idx)
                            bin_assignments[src_bin].remove(box_idx)
                            if not bin_assignments[src_bin]:
                                del bin_assignments[src_bin]
                                bin_ids.remove(src_bin)
                            
                            placed_elsewhere = True
                            improved = True
                            break
                    
                    if placed_elsewhere:
                        # Restart the consolidation loop
                        bin_ids = sorted(bin_assignments.keys())
                        break
                
                if improved:
                    break
        
        # Recompute final score after consolidation
        if best_placements:
            bin_ids_final = set(p[0] for p in best_placements)
            B_final = len(bin_ids_final)
            last_bin_id = max(bin_ids_final)
            
            # Calculate last bin fill
            last_bin_vol = 0
            for p in best_placements:
                if p[0] == last_bin_id:
                    last_bin_vol += p[4] * p[5] * p[6]
            
            fill_final = last_bin_vol / container_vol
            final_score = B_final + 0.1 * (1 - fill_final)
    
    return best_placements
