"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Build a list of unique dimensions for grid granularity
    all_dims = set()
    for _, w, h, d, _ in indexed_boxes:
        all_dims.add(w)
        all_dims.add(h)
        all_dims.add(d)
    
    # We'll use a grid-based approach with a resolution that's the GCD of all dimensions
    # But to keep it simple and fast, we'll use a voxel grid at unit resolution if feasible
    # Otherwise, we'll use a coarser grid or fall back to pairwise checks
    
    # For now, let's use a hybrid: occupancy tracking via list of placed boxes per bin
    # but with a spatial hash for faster overlap checks
    
    def create_bin():
        return {
            'placements': [],  # list of (x, y, z, w, h, d)
            'grid': None,  # numpy array if used
            'used_vol': 0
        }
    
    def can_place_in_bin(bin_state, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_state['placements']:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_spot(bin_state, w, h, d):
        """Find the best spot for a box in a bin."""
        placements = bin_state['placements']
        
        # Generate candidate positions
        candidates = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter and score candidates
        valid = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place_in_bin(bin_state, x, y, z, w, h, d):
                    valid.append((x, y, z))
        
        if not valid:
            return None
        
        # Prefer positions that minimize z, then y, then x (bottom-left-back)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid[0]
    
    def pack_ffd():
        """First-Fit-Decreasing packing."""
        bins = []
        placements = [None] * n
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
            placed = False
            for b_idx in range(len(bins)):
                bin_state = bins[b_idx]
                spot = find_spot(bin_state, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    bin_state['placements'].append((x, y, z, w, h, d))
                    bin_state['used_vol'] += vol
                    placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                b_idx = len(bins)
                bin_state = create_bin()
                bin_state['placements'].append((0, 0, 0, w, h, d))
                bin_state['used_vol'] += vol
                bins.append(bin_state)
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        
        return placements, bins
    
    def score_placements(placements, bins):
        """Calculate the score."""
        if not placements or not bins:
            return float('inf')
        
        # Number of occupied bins
        occupied_bins = set()
        for p in placements:
            if p is not None:
                occupied_bins.add(p[0])
        B = len(occupied_bins)
        
        # Last bin is the largest label
        last_bin = max(occupied_bins) if occupied_bins else 0
        
        # Volume in last bin
        last_bin_vol = 0
        for p in placements:
            if p is not None and p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        if container_vol == 0:
            return float(B)
        
        last_bin_fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - last_bin_fraction)
    
    # Initial FFD packing
    best_placements, best_bins = pack_ffd()
    best_score = score_placements(best_placements, best_bins)
    
    # Local search: try to reduce number of bins and improve last bin fill
    def try_move_box(placements, bins, box_idx, target_bin_idx):
        """Try to move box_idx to target_bin_idx. Returns True if successful."""
        b, x, y, z, w, h, d = placements[box_idx]
        if target_bin_idx == b:
            return False
        
        # Remove from current bin
        current_bin = bins[b]
        # Find and remove the placement
        for i, p in enumerate(current_bin['placements']):
            if p == (x, y, z, w, h, d):
                current_bin['placements'].pop(i)
                current_bin['used_vol'] -= w * h * d
                break
        
        # Try to place in target bin
        target_bin = bins[target_bin_idx]
        spot = find_spot(target_bin, w, h, d)
        if spot is not None:
            tx, ty, tz = spot
            target_bin['placements'].append((tx, ty, tz, w, h, d))
            target_bin['used_vol'] += w * h * d
            placements[box_idx] = (target_bin_idx, tx, ty, tz, w, h, d)
            return True
        else:
            # Revert
            current_bin['placements'].append((x, y, z, w, h, d))
            current_bin['used_vol'] += w * h * d
            placements[box_idx] = (b, x, y, z, w, h, d)
            return False
    
    def try_swap_boxes(placements, bins, box_i, box_j):
        """Try to swap boxes i and j between their bins."""
        bi, xi, yi, zi, wi, hi, di = placements[box_i]
        bj, xj, yj, zj, wj, hj, dj = placements[box_j]
        
        if bi == bj:
            return False
        
        # Remove both from their bins
        bin_i = bins[bi]
        bin_j = bins[bj]
        
        # Find and remove box_i from bin_i
        for k, p in enumerate(bin_i['placements']):
            if p == (xi, yi, zi, wi, hi, di):
                bin_i['placements'].pop(k)
                bin_i['used_vol'] -= wi * hi * di
                break
        
        # Find and remove box_j from bin_j
        for k, p in enumerate(bin_j['placements']):
            if p == (xj, yj, zj, wj, hj, dj):
                bin_j['placements'].pop(k)
                bin_j['used_vol'] -= wj * hj * dj
                break
        
        # Try to place box_i in bin_j
        spot_i = find_spot(bin_j, wi, hi, di)
        if spot_i is None:
            # Revert
            bin_i['placements'].append((xi, yi, zi, wi, hi, di))
            bin_i['used_vol'] += wi * hi * di
            bin_j['placements'].append((xj, yj, zj, wj, hj, dj))
            bin_j['used_vol'] += wj * hj * dj
            return False
        
        # Try to place box_j in bin_i
        spot_j = find_spot(bin_i, wj, hj, dj)
        if spot_j is None:
            # Revert
            bin_i['placements'].append((xi, yi, zi, wi, hi, di))
            bin_i['used_vol'] += wi * hi * di
            bin_j['placements'].append((xj, yj, zj, wj, hj, dj))
            bin_j['used_vol'] += wj * hj * dj
            return False
        
        # Apply swap
        bin_j['placements'].append((spot_i[0], spot_i[1], spot_i[2], wi, hi, di))
        bin_j['used_vol'] += wi * hi * di
        bin_i['placements'].append((spot_j[0], spot_j[1], spot_j[2], wj, hj, dj))
        bin_i['used_vol'] += wj * hj * dj
        
        placements[box_i] = (bj, spot_i[0], spot_i[1], spot_i[2], wi, hi, di)
        placements[box_j] = (bi, spot_j[0], spot_j[1], spot_j[2], wj, hj, dj)
        
        return True
    
    # Local search iterations
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        max_iterations = 500
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try moving each box to another bin
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                if best_placements[i] is None:
                    continue
                
                orig_bin = best_placements[i][0]
                
                # Try moving to each other bin
                for target_bin in range(len(best_bins)):
                    if target_bin == orig_bin:
                        continue
                    
                    if try_move_box(best_placements, best_bins, i, target_bin):
                        new_score = score_placements(best_placements, best_bins)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            try_move_box(best_placements, best_bins, i, orig_bin)
                
                if improved:
                    break
            
            # Try swapping boxes
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    if best_placements[i] is None:
                        continue
                    for j in range(i + 1, n):
                        if time.time() > deadline - 10:
                            break
                        if best_placements[j] is None:
                            continue
                        
                        if try_swap_boxes(best_placements, best_bins, i, j):
                            new_score = score_placements(best_placements, best_bins)
                            if new_score < best_score:
                                best_score = new_score
                                improved = True
                                break
                            else:
                                # Revert swap
                                try_swap_boxes(best_placements, best_bins, i, j)
                        
                        if improved:
                            break
                    if improved:
                        break
    
    # Verify validity
    def is_valid(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        # Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True
    
    if not is_valid(best_placements):
        # Fallback: simple shelf packing
        best_placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return best_placements
# EVOLVE-BLOCK-END
