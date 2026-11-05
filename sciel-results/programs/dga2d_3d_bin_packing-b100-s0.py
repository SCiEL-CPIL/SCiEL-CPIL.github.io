"""Rewritten 3D bin packing with multi-strategy local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    total_vol = sum(volumes)
    container_vol = W * H * D
    
    # Sort indices by different criteria
    by_vol = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    by_max_dim = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    by_surface = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    strategies = [
        ('vol', by_vol),
        ('maxdim', by_max_dim),
        ('surface', by_surface),
    ]
    
    def fits(occupied, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with occupied boxes
        for (ox, oy, oz, ow, oh, od) in occupied:
            if x < ox + ow and ox < x + w and \
               y < oy + oh and oy < y + h and \
               z < oz + od and oz < z + d:
                return False
        return True
    
    def find_position(occupied, w, h, d):
        """Find a position for the box, preferring lower z, then y, then x."""
        # Try to find the best (lowest z, then y, then x) position
        best_pos = None
        best_key = None
        
        # Sample candidate positions - check corners and existing box boundaries
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (ox, oy, oz, ow, oh, od) in occupied:
            if ox + ow <= W:
                x_candidates.append(ox + ow)
            if oy + oh <= H:
                y_candidates.append(oy + oh)
            if oz + od <= D:
                z_candidates.append(oz + od)
        
        # Also add some random candidates for diversity
        for _ in range(min(5, len(occupied))):
            x_candidates.append(random.randint(0, W - w))
            y_candidates.append(random.randint(0, H - h))
            z_candidates.append(random.randint(0, D - d))
        
        x_candidates = sorted(set(x for x in x_candidates if 0 <= x <= W - w))
        y_candidates = sorted(set(y for y in y_candidates if 0 <= y <= H - h))
        z_candidates = sorted(set(z for z in z_candidates if 0 <= z <= D - d))
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if fits(occupied, x, y, z, w, h, d):
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
                        break
                if best_pos is not None and best_key[0] == z:
                    # Found something at this z, could break early for speed
                    # but let's keep looking for better y,x at same z
                    pass
        
        return best_pos
    
    def greedy_pack(order, occupied_lists, placements_by_bin):
        """Place boxes in given order into bins."""
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            # Try to place in existing bins first
            for b in range(len(occupied_lists)):
                pos = find_position(occupied_lists[b], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    occupied_lists[b].append((x, y, z, w, h, d))
                    placements_by_bin[idx] = (b, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                # Try to open a new bin
                b = len(occupied_lists)
                occupied_lists.append([])
                # Place at origin
                occupied_lists[b].append((0, 0, 0, w, h, d))
                placements_by_bin[idx] = (b, 0, 0, 0, w, h, d)
        
        return placements_by_bin
    
    def compute_score(placements_by_bin):
        """Compute the score: B + 0.1*(1 - last_bin_occupied/container_vol)."""
        if not placements_by_bin:
            return 0.0
        
        bins_used = set()
        for idx in range(n):
            b, x, y, z, w, h, d = placements_by_bin[idx]
            bins_used.add(b)
        
        B = len(bins_used)
        
        # Find the last bin (largest label)
        last_bin = max(bins_used)
        
        # Compute occupied volume in last bin
        last_bin_vol = 0
        for idx in range(n):
            b, x, y, z, w, h, d = placements_by_bin[idx]
            if b == last_bin:
                last_bin_vol += w * h * d
        
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - occupied_frac)
        return score
    
    def local_search(placements_by_bin, occupied_lists, max_iters=50):
        """Try to improve by moving boxes between bins or to better positions."""
        current_score = compute_score(placements_by_bin)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try moving each box to a better position in its current bin or another bin
            for idx in range(n):
                b, x, y, z, w, h, d = placements_by_bin[idx]
                
                # Remove box from current bin
                occupied_lists[b].remove((x, y, z, w, h, d))
                
                best_new_score = current_score
                best_new_placement = None
                best_new_bin = b
                best_new_pos = (x, y, z)
                
                # Try all bins
                for target_b in range(len(occupied_lists)):
                    pos = find_position(occupied_lists[target_b], w, h, d)
                    if pos is not None:
                        nx, ny, nz = pos
                        # Tentatively place
                        occupied_lists[target_b].append((nx, ny, nz, w, h, d))
                        placements_by_bin[idx] = (target_b, nx, ny, nz, w, h, d)
                        new_score = compute_score(placements_by_bin)
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_placement = (target_b, nx, ny, nz, w, h, d)
                            best_new_bin = target_b
                            best_new_pos = (nx, ny, nz)
                        else:
                            # Revert
                            occupied_lists[target_b].remove((nx, ny, nz, w, h, d))
                
                if best_new_placement is not None and best_new_score < current_score:
                    # Apply the best move
                    if best_new_bin != b:
                        # Box was removed from bin b, add to new bin
                        occupied_lists[best_new_bin].append(best_new_pos + (w, h, d))
                    else:
                        occupied_lists[b].append(best_new_pos + (w, h, d))
                    placements_by_bin[idx] = best_new_placement
                    current_score = best_new_score
                    improved = True
                else:
                    # Revert to original
                    occupied_lists[b].append((x, y, z, w, h, d))
                    placements_by_bin[idx] = (b, x, y, z, w, h, d)
            
            if not improved:
                break
        
        return current_score
    
    best_score = float('inf')
    best_placements = None
    
    for strat_name, order in strategies:
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts for each strategy
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Shuffle the order slightly for diversity
            if r > 0:
                # Perturb the order
                order_copy = order[:]
                for _ in range(min(5, n // 2)):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    order_copy[i], order_copy[j] = order_copy[j], order_copy[i]
                order = order_copy
            
            occupied_lists = []
            placements_by_bin = {}
            
            greedy_pack(order, occupied_lists, placements_by_bin)
            
            if not placements_by_bin:
                continue
            
            score = local_search(placements_by_bin, occupied_lists)
            
            if score < best_score:
                best_score = score
                best_placements = dict(placements_by_bin)
                best_occupied = [list(b) for b in occupied_lists]
    
    # Build final result
    if best_placements is None:
        # Fallback: simple shelf packing
        W, H, D = container
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
        return placements
    
    result = []
    for i in range(n):
        result.append(best_placements[i])
    
    return result

# EVOLVE-BLOCK-END
