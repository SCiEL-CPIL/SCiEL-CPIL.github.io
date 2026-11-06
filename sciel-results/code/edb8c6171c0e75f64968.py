import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending for initial packing
    sorted_indices = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def pack_with_order(order):
        """Pack boxes in given order using shelf-based 3D packing."""
        bins = []  # list of (x, y, z, w, h, d) tuples per bin
        bin_shelves = []  # list of list of (x, y, z, row_h, layer_d) per bin
        placements = []
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            for b in range(len(bins)):
                # Try to place in existing bin b
                shelves = bin_shelves[b]
                
                # Try each existing shelf
                for s in range(len(shelves)):
                    sx, sy, sz, row_h, layer_d = shelves[s]
                    # Check if box fits on this shelf
                    if sx + w <= W and sy + h <= H and sz + d <= D:
                        # Check overlap with existing boxes in this bin
                        can_place = True
                        for (bx, by, bz, bw, bh, bd) in bins[b]:
                            # Check if new box overlaps with existing box
                            if not (sx + w <= bx or bx + bw <= sx or
                                    sy + h <= by or by + bh <= sy or
                                    sz + d <= bz or bz + bd <= sz):
                                can_place = False
                                break
                        if can_place:
                            bins[b].append((sx, sy, sz, w, h, d))
                            shelves[s] = (sx + w, sy, sz, max(row_h, h), max(layer_d, d))
                            placements.append((b, sx, sy, sz, w, h, d))
                            placed = True
                            break
                
                if placed:
                    break
                
                # Try to create a new shelf in this bin
                # Find the lowest z where we can start a new shelf
                # Check all existing shelves to find gaps
                if not placed:
                    # Try starting a new shelf at various z positions
                    # First, try at z=0 (bottom)
                    can_new_shelf = True
                    for (bx, by, bz, bw, bh, bd) in bins[b]:
                        if bz + bd > 0 and bz < H:
                            # There's something above z=0, so we can't start at z=0 unless it's empty
                            pass
                    
                    # Simpler: try to place at (0, 0, z) for various z
                    # Find all z-levels where we might start a new shelf
                    z_levels = [0]
                    for (bx, by, bz, bw, bh, bd) in bins[b]:
                        if bz + bd <= H:
                            z_levels.append(bz + bd)
                    z_levels = sorted(set(z_levels))
                    
                    for z_new in z_levels:
                        if z_new + d > H:
                            continue
                        # Check if we can place at (0, 0, z_new) without overlap
                        can_place = True
                        for (bx, by, bz, bw, bh, bd) in bins[b]:
                            if not (0 + w <= bx or bx + bw <= 0 or
                                    0 + h <= by or by + bh <= 0 or
                                    z_new + d <= bz or bz + bd <= z_new):
                                can_place = False
                                break
                        if can_place:
                            bins[b].append((0, 0, z_new, w, h, d))
                            bin_shelves[b].append((w, 0, z_new, h, d))
                            placements.append((b, 0, 0, z_new, w, h, d))
                            placed = True
                            break
            
            if not placed:
                # Start a new bin
                b = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_shelves.append([(w, 0, 0, h, d)])
                placements.append((b, 0, 0, 0, w, h, d))
        
        return placements
    
    def score_placements(placements):
        """Calculate the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0.0
        
        # Find all bin labels used
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        
        if B == 0:
            return 0.0
        
        # Find the last bin (largest label)
        last_bin = max(bin_labels)
        
        # Calculate occupied volume in last bin
        last_bin_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        last_bin_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - last_bin_fraction)
        return score
    
    # Initial packing with sorted order
    initial_order = sorted_indices
    best_placements = pack_with_order(initial_order)
    best_score = score_placements(best_placements)
    
    # Local search: try swapping pairs of boxes in the order
    current_order = list(initial_order)
    
    # Also try some random restarts with different orderings
    num_restarts = 0
    max_restarts = 50
    num_swaps_per_restart = 20
    
    while time.time() < deadline - 5.0:  # 5s margin for final evaluation
        # Try local search on current best
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 5.0:
                    break
                # Swap boxes at positions i and j
                current_order[i], current_order[j] = current_order[j], current_order[i]
                placements = pack_with_order(current_order)
                score = score_placements(placements)
                
                if score < best_score:
                    best_score = score
                    best_placements = placements
                    improved = True
                else:
                    # Swap back
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        if not improved and num_restarts < max_restarts:
            # Random restart
            random.shuffle(current_order)
            placements = pack_with_order(current_order)
            score = score_placements(placements)
            if score < best_score:
                best_score = score
                best_placements = placements
            num_restarts += 1
        
        # If we've done many restarts without improvement, stop
        if num_restarts >= max_restarts:
            break
    
    return best_placements
