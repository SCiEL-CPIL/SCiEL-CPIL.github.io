"""3D Bin Packing: parameterized placement with local search optimization."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def can_place(placed, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if is_overlap(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def get_valid_positions(placed, w, h, d):
        """Generate candidate positions using face extensions and corners."""
        cands = set()
        cands.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            # Face extensions
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            # Corner combinations
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
            
        valid = []
        for (cx, cy, cz) in cands:
            if can_place(placed, cx, cy, cz, w, h, d):
                valid.append((cx, cy, cz))
        
        # Sort by compactness (distance to origin) to prefer tight packing
        valid.sort(key=lambda c: c[0] + c[1] + c[2])
        return valid

    def score_assignment(assignments):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    # --- Data Structure: bins as list of lists of (idx, x, y, z, w, h, d) ---
    
    def greedy_pack(order, bins, assignments):
        """Place boxes in given order using best-fit decreasing style."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed_in_bin = False
            
            if bins:
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    if not bins[b_id]:
                        continue
                    
                    # Extract placements for overlap check
                    placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[b_id]]
                    cands = get_valid_positions(placements, w, h, d)
                    if cands:
                        cx, cy, cz = cands[0]
                        metric = cx + cy + cz
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].append((idx, cx, cy, cz, w, h, d))
                    assignments[idx] = best_bin_id
                    placed_in_bin = True
            
            if not placed_in_bin:
                b_id = len(bins)
                bins.append([(idx, 0, 0, 0, w, h, d)])
                assignments[idx] = b_id

    def local_search(bins, assignments):
        """Local search to minimize bins and maximize last-bin fill."""
        if time.time() > deadline:
            return

        improved = True
        iter_count = 0
        max_iters = 2000
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Phase 1: Reduce B by moving boxes from higher bins to lower bins
            for b_id in range(len(bins) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                
                boxes_in_bin = list(bins[b_id])
                for local_idx in range(len(boxes_in_bin)):
                    if time.time() > deadline:
                        break
                    
                    idx, px, py, pz, pw, ph, pd = boxes_in_bin[local_idx]
                    
                    # Verify the box is still in this bin
                    if not any(b[0] == idx for b in bins[b_id]):
                        continue
                    
                    # Try to remove and place in an earlier bin
                    bins[b_id].remove((idx, px, py, pz, pw, ph, pd))
                    
                    moved = False
                    for target_b in range(b_id):
                        if not bins[target_b]:
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[target_b]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[target_b].append((idx, cx, cy, cz, pw, ph, pd))
                            assignments[idx] = target_b
                            moved = True
                            improved = True
                            break
                    
                    if not moved:
                        # Put it back
                        bins[b_id].append((idx, px, py, pz, pw, ph, pd))
                        assignments[idx] = b_id
        
        # Phase 2: Try to fill the last bin more
        if time.time() < deadline:
            occupied_bins = [b for b in range(len(bins)) if bins[b]]
            if occupied_bins:
                last_bin_id = max(occupied_bins)
                
                for b_id in sorted(occupied_bins, reverse=True):
                    if time.time() > deadline:
                        break
                    if b_id == last_bin_id or not bins[b_id]:
                        continue
                    
                    boxes_in_bin = list(bins[b_id])
                    for local_idx in range(len(boxes_in_bin)):
                        if time.time() > deadline:
                            break
                        
                        idx, px, py, pz, pw, ph, pd = boxes_in_bin[local_idx]
                        
                        if not any(b[0] == idx for b in bins[b_id]):
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[last_bin_id]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[b_id].remove((idx, px, py, pz, pw, ph, pd))
                            bins[last_bin_id].append((idx, cx, cy, cz, pw, ph, pd))
                            assignments[idx] = last_bin_id
                            improved = True

    def build_result(bins, assignments):
        """Construct result list from bins and assignments."""
        result = [None] * n
        for idx in range(n):
            if idx in assignments:
                b_id = assignments[idx]
                for (i, x, y, z, w, h, d) in bins[b_id]:
                    if i == idx:
                        result[idx] = (b_id, x, y, z, w, h, d)
                        break
                if result[idx] is None:
                    # Fallback
                    result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            else:
                # Box wasn't placed - create new bin
                b_id = len(bins)
                bins.append([(idx, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])])
                assignments[idx] = b_id
                result[idx] = (b_id, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
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

    # --- CSP-based local search with forward checking ---
    def csp_local_search(bins, assignments):
        """CSP-inspired local search using interval-based conflict detection."""
        if time.time() > deadline:
            return

        # Build spatial index per bin for faster overlap checks
        bin_indices = []
        for b_id in range(len(bins)):
            if not bins[b_id]:
                bin_indices.append(None)
                continue
            # Build list of (x1, x2, y1, y2, z1, z2, box_tuple)
            indexed = []
            for (idx, x, y, z, w, h, d) in bins[b_id]:
                indexed.append((x, x+w, y, y+h, z, z+d, (idx, x, y, z, w, h, d)))
            bin_indices.append(indexed)
        
        def fast_can_place(bin_idx, x, y, z, w, h, d):
            """Fast overlap check using pre-built index."""
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            if bin_idx is None:
                return True
            for (bx1, bx2, by1, by2, bz1, bz2, _) in bin_idx:
                if not (x + w <= bx1 or bx1 + 0 <= x or  # simplified: use original logic
                        x + w <= bx1 or bx2 <= x or
                        y + h <= by1 or by2 <= y or
                        z + d <= bz1 or bz2 <= z):
                    # More precise check
                    if not (x + w <= bx1 or bx2 <= x or
                            y + h <= by1 or by2 <= y or
                            z + d <= bz1 or bz2 <= z):
                        return False
            return True

        def rebuild_bin_index(b_id):
            if not bins[b_id]:
                bin_indices[b_id] = None
                return
            indexed = []
            for (idx, x, y, z, w, h, d) in bins[b_id]:
                indexed.append((x, x+w, y, y+h, z, z+d, (idx, x, y, z, w, h, d)))
            bin_indices[b_id] = indexed

        improved = True
        iter_count = 0
        max_iters = 500
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Phase 1: Reduce B by moving boxes from higher bins to lower bins
            for b_id in range(len(bins) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                
                boxes_in_bin = list(bins[b_id])
                for local_idx in range(len(boxes_in_bin)):
                    if time.time() > deadline:
                        break
                    
                    idx, px, py, pz, pw, ph, pd = boxes_in_bin[local_idx]
                    
                    # Verify the box is still in this bin
                    if not any(b[0] == idx for b in bins[b_id]):
                        continue
                    
                    # Remove from current bin
                    bins[b_id].remove((idx, px, py, pz, pw, ph, pd))
                    rebuild_bin_index(b_id)
                    
                    moved = False
                    for target_b in range(b_id):
                        if not bins[target_b]:
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[target_b]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[target_b].append((idx, cx, cy, cz, pw, ph, pd))
                            rebuild_bin_index(target_b)
                            assignments[idx] = target_b
                            moved = True
                            improved = True
                            break
                    
                    if not moved:
                        # Put it back
                        bins[b_id].append((idx, px, py, pz, pw, ph, pd))
                        rebuild_bin_index(b_id)
                        assignments[idx] = b_id
        
        # Phase 2: Try to fill the last bin more
        if time.time() < deadline:
            occupied_bins = [b for b in range(len(bins)) if bins[b]]
            if occupied_bins:
                last_bin_id = max(occupied_bins)
                
                for b_id in sorted(occupied_bins, reverse=True):
                    if time.time() > deadline:
                        break
                    if b_id == last_bin_id or not bins[b_id]:
                        continue
                    
                    boxes_in_bin = list(bins[b_id])
                    for local_idx in range(len(boxes_in_bin)):
                        if time.time() > deadline:
                            break
                        
                        idx, px, py, pz, pw, ph, pd = boxes_in_bin[local_idx]
                        
                        if not any(b[0] == idx for b in bins[b_id]):
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[last_bin_id]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[b_id].remove((idx, px, py, pz, pw, ph, pd))
                            rebuild_bin_index(b_id)
                            bins[last_bin_id].append((idx, cx, cy, cz, pw, ph, pd))
                            rebuild_bin_index(last_bin_id)
                            assignments[idx] = last_bin_id
                            improved = True

    # --- Main search loop ---
    
    strategies = [
        lambda t: -(t[1][0] * t[1][1] * t[1][2]),  # Volume desc
        lambda t: t[1][0] * t[1][1] * t[1][2],      # Volume asc
        lambda t: -(max(t[1])),                      # Max dim desc
        lambda t: -(t[1][0] + t[1][1] + t[1][2]),   # Sum dim desc
        lambda t: -(min(t[1])),                      # Min dim desc
        lambda t: -(t[1][0] * t[1][1]),              # Area desc
        lambda t: -(t[1][0] * t[1][2]),              # XZ area desc
        lambda t: -(t[1][1] * t[1][2]),              # YZ area desc
    ]

    best_score = float('inf')
    best_result = None

    max_restarts = 40
    
    for strat_idx, strat_func in enumerate(strategies):
        if time.time() > deadline:
            break
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            order = list(enumerate(boxes))
            
            if restart == 0:
                order.sort(key=strat_func)
            else:
                # Add noise for diversity
                order.sort(key=lambda t: strat_func(t) + random.uniform(-0.3, 0.3) * (t[1][0] * t[1][1] * t[1][2] + 1))
            
            bins = []
            assignments = {}
            
            greedy_pack(order, bins, assignments)
            
            # Local search to improve
            local_search(bins, assignments)
            
            # CSP-based local search for further improvement
            csp_local_search(bins, assignments)
            
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
