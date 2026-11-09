"""3D Bin Packing: improved placement with better local search and packing strategies."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    
    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def bin_can_place(placed, x, y, z, w, h, d):
        """Check if box (x,y,z,w,h,d) fits in a bin with existing placements."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if is_overlap(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def get_candidate_positions(placed, w, h, d):
        """Generate candidate positions for placing a box in a bin using skyline approach."""
        cands = set()
        cands.add((0, 0, 0))
        
        # Generate positions based on existing boxes
        for (px, py, pz, pw, ph, pd) in placed:
            # Positions adjacent to existing boxes
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
        
        valid = []
        for (cx, cy, cz) in cands:
            if bin_can_place(placed, cx, cy, cz, w, h, d):
                valid.append((cx, cy, cz))
        
        return valid

    def score_assignment(assignments, box_vols, container_vol):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    class BinState:
        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
        
        def is_empty(self):
            return len(self.boxes) == 0
        
        def get_placements(self):
            return [(x, y, z, w, h, d) for (_, x, y, z, w, h, d) in self.boxes]
        
        def remove_box(self, idx):
            """Remove box with given index, return it."""
            for i, box in enumerate(self.boxes):
                if box[0] == idx:
                    return self.boxes.pop(i)
            return None

    def greedy_pack(order, bins, assignments, strategy='best_fit'):
        """Place boxes in given order using specified strategy."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed_in_bin = False
            
            if bins:
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    if bins[b_id].is_empty():
                        continue
                    
                    cands = get_candidate_positions(bins[b_id].get_placements(), w, h, d)
                    if cands:
                        if strategy == 'best_fit':
                            # Minimize distance from origin (compactness)
                            cands.sort(key=lambda c: c[0] + c[1] + c[2])
                            cx, cy, cz = cands[0]
                            metric = cx + cy + cz
                        elif strategy == 'first_fit':
                            cx, cy, cz = cands[0]
                            metric = 0
                        elif strategy == 'best_fit_decreasing':
                            # Prefer positions that leave more space
                            cands.sort(key=lambda c: -(c[0] + c[1] + c[2]))
                            cx, cy, cz = cands[0]
                            metric = cx + cy + cz
                        else:
                            cands.sort(key=lambda c: c[0] + c[1] + c[2])
                            cx, cy, cz = cands[0]
                            metric = cx + cy + cz
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].boxes.append((idx, cx, cy, cz, w, h, d))
                    assignments[idx] = best_bin_id
                    placed_in_bin = True
            
            if not placed_in_bin:
                b_id = len(bins)
                bins.append(BinState())
                bins[b_id].boxes.append((idx, 0, 0, 0, w, h, d))
                assignments[idx] = b_id

    def local_search(bins, assignments, max_iters=2000):
        """Local search to minimize bins and maximize last-bin fill."""
        if time.time() > deadline:
            return

        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to move boxes from higher-numbered bins to lower-numbered bins
            occupied_bins = sorted([b for b in range(len(bins)) if not bins[b].is_empty()], reverse=True)
            
            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if bins[b_id].is_empty():
                    continue
                
                # Get boxes in this bin (make a copy to avoid modification during iteration)
                boxes_in_bin = list(bins[b_id].boxes)
                
                for (idx, px, py, pz, pw, ph, pd) in boxes_in_bin:
                    if time.time() > deadline:
                        break
                    
                    # Try to remove and place in an earlier bin
                    removed = bins[b_id].remove_box(idx)
                    if removed is None:
                        continue
                    
                    moved = False
                    for target_b in range(b_id):
                        if bins[target_b].is_empty():
                            continue
                        
                        cands = get_candidate_positions(bins[target_b].get_placements(), pw, ph, pd)
                        if cands:
                            cands.sort(key=lambda c: c[0] + c[1] + c[2])
                            cx, cy, cz = cands[0]
                            bins[target_b].boxes.append((idx, cx, cy, cz, pw, ph, pd))
                            assignments[idx] = target_b
                            moved = True
                            improved = True
                            break
                    
                    if not moved:
                        # Put it back
                        bins[b_id].boxes.append((idx, px, py, pz, pw, ph, pd))
                        assignments[idx] = b_id
            
            # Phase 2: Try to fill the last bin more by moving boxes into it
            if time.time() < deadline:
                occupied_bins = [b for b in range(len(bins)) if not bins[b].is_empty()]
                if occupied_bins:
                    last_bin_id = max(occupied_bins)
                    
                    for b_id in sorted(occupied_bins, reverse=True):
                        if time.time() > deadline:
                            break
                        if b_id == last_bin_id or bins[b_id].is_empty():
                            continue
                        
                        boxes_in_bin = list(bins[b_id].boxes)
                        for (idx, px, py, pz, pw, ph, pd) in boxes_in_bin:
                            if time.time() > deadline:
                                break
                            
                            cands = get_candidate_positions(bins[last_bin_id].get_placements(), pw, ph, pd)
                            if cands:
                                cands.sort(key=lambda c: c[0] + c[1] + c[2])
                                cx, cy, cz = cands[0]
                                bins[b_id].remove_box(idx)
                                bins[last_bin_id].boxes.append((idx, cx, cy, cz, pw, ph, pd))
                                assignments[idx] = last_bin_id
                                improved = True
            
            # Phase 3: Try to swap boxes between bins to improve packing
            if time.time() < deadline:
                occupied_bins = [b for b in range(len(bins)) if not bins[b].is_empty()]
                if len(occupied_bins) >= 2:
                    for b1 in occupied_bins:
                        if time.time() > deadline:
                            break
                        for b2 in occupied_bins:
                            if time.time() > deadline:
                                break
                            if b1 >= b2:
                                continue
                            
                            boxes1 = list(bins[b1].boxes)
                            boxes2 = list(bins[b2].boxes)
                            
                            for (idx1, x1, y1, z1, w1, h1, d1) in boxes1:
                                if time.time() > deadline:
                                    break
                                for (idx2, x2, y2, z2, w2, h2, d2) in boxes2:
                                    if time.time() > deadline:
                                        break
                                    
                                    # Try swapping: remove both, try to place in other's bin
                                    bins[b1].remove_box(idx1)
                                    bins[b2].remove_box(idx2)
                                    
                                    cands1 = get_candidate_positions(bins[b2].get_placements(), w1, h1, d1)
                                    cands2 = get_candidate_positions(bins[b1].get_placements(), w2, h2, d2)
                                    
                                    if cands1 and cands2:
                                        cands1.sort(key=lambda c: c[0] + c[1] + c[2])
                                        cands2.sort(key=lambda c: c[0] + c[1] + c[2])
                                        cx1, cy1, cz1 = cands1[0]
                                        cx2, cy2, cz2 = cands2[0]
                                        bins[b2].boxes.append((idx1, cx1, cy1, cz1, w1, h1, d1))
                                        bins[b1].boxes.append((idx2, cx2, cy2, cz2, w2, h2, d2))
                                        assignments[idx1] = b2
                                        assignments[idx2] = b1
                                        improved = True
                                    else:
                                        # Put back
                                        bins[b1].boxes.append((idx1, x1, y1, z1, w1, h1, d1))
                                        bins[b2].boxes.append((idx2, x2, y2, z2, w2, h2, d2))
                                        assignments[idx1] = b1
                                        assignments[idx2] = b2

    def build_result(bins, assignments):
        """Construct result list from bins and assignments."""
        result = [None] * n
        for idx in range(n):
            if idx in assignments:
                b_id = assignments[idx]
                for (i, x, y, z, w, h, d) in bins[b_id].boxes:
                    if i == idx:
                        result[idx] = (b_id, x, y, z, w, h, d)
                        break
                if result[idx] is None:
                    # Fallback - shouldn't happen
                    result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            else:
                # Box wasn't placed - create new bin
                b_id = len(bins)
                bins.append(BinState())
                w, h, d = boxes[idx]
                bins[b_id].boxes.append((idx, 0, 0, 0, w, h, d))
                assignments[idx] = b_id
                result[idx] = (b_id, 0, 0, 0, w, h, d)
        
        # Renumber bins to be consecutive starting from 0
        if result:
            bin_labels = sorted(set(r[0] for r in result if r is not None))
            if bin_labels:
                label_map = {old: new for new, old in enumerate(bin_labels)}
                for i in range(n):
                    if result[i] is not None:
                        old_bin = result[i][0]
                        new_bin = label_map[old_bin]
                        result[i] = (new_bin, result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6])
        
        return result

    def validate_result(result):
        """Validate that result is feasible."""
        bins_boxes = {}
        for idx in range(n):
            if result[idx] is None:
                return False
            b, x, y, z, w, h, d = result[idx]
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            if b not in bins_boxes:
                bins_boxes[b] = []
            bins_boxes[b].append((x, y, z, w, h, d))
        
        for b, bl in bins_boxes.items():
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                                  bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                        return False
        return True

    def score_from_result(result):
        """Compute score from result."""
        if not result:
            return float('inf')
        bins_used = set(r[0] for r in result)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(r[4] * r[5] * r[6] for r in result if r[0] == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    # --- Main search loop with multiple strategies ---
    
    strategies = [
        ('vol_desc', lambda t: -(t[1][0] * t[1][1] * t[1][2])),
        ('vol_asc', lambda t: t[1][0] * t[1][1] * t[1][2]),
        ('max_dim_desc', lambda t: -(max(t[1]))),
        ('sum_dim_desc', lambda t: -(t[1][0] + t[1][1] + t[1][2])),
        ('min_dim_desc', lambda t: -(min(t[1]))),
        ('area_desc', lambda t: -(t[1][0] * t[1][1])),
        ('area_asc', lambda t: t[1][0] * t[1][1]),
        ('perimeter_desc', lambda t: -(2*(t[1][0] + t[1][1]))),
    ]

    best_score = float('inf')
    best_result = None

    max_restarts_per_strategy = 10
    
    for strat_name, strat_func in strategies:
        if time.time() > deadline:
            break
        
        for restart in range(max_restarts_per_strategy):
            if time.time() > deadline:
                break
            
            order = list(enumerate(boxes))
            
            if restart == 0:
                order.sort(key=strat_func)
            else:
                # Add noise for diversity
                noise_scale = 0.1 * (max(box_vols) + 1)
                order.sort(key=lambda t: strat_func(t) + random.uniform(-noise_scale, noise_scale))
            
            bins = []
            assignments = {}
            
            # Use different strategies for different restarts
            if restart % 3 == 0:
                strategy = 'best_fit'
            elif restart % 3 == 1:
                strategy = 'first_fit'
            else:
                strategy = 'best_fit_decreasing'
            
            greedy_pack(order, bins, assignments, strategy)
            
            # Local search to improve
            local_search(bins, assignments, 500)
            
            result = build_result(bins, assignments)
            
            # Validate
            if not validate_result(result):
                continue
            
            # Compute score
            score = score_from_result(result)
            
            if score < best_score:
                best_score = score
                best_result = result
    
    # If no valid result found, use fallback
    if best_result is None:
        # Simple shelf packing fallback
        placements = {}
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
        
        for idx, (w, h, d) in indexed_boxes:
            # Try to place in current position
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
                if y + h > H:
                    y = 0
                    z += layer_d
                    layer_d = 0
                    x, row_h = 0, 0
                    if z + d > D:
                        b += 1
                        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
            
            if x + w > W or y + h > H or z + d > D:
                b += 1
                x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
            
            placements[idx] = (b, x, y, z, w, h, d)
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        
        best_result = [None] * n
        for idx in range(n):
            b, x, y, z, w, h, d = placements[idx]
            best_result[idx] = (b, x, y, z, w, h, d)
    
    return best_result
# EVOLVE-BLOCK-END
