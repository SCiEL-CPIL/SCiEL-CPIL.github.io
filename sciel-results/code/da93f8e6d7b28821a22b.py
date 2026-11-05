"""3D bin packing with simulated annealing local search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
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
        for (ox, oy, oz, ow, oh, od) in occupied:
            if x < ox + ow and ox < x + w and \
               y < oy + oh and oy < y + h and \
               z < oz + od and oz < z + d:
                return False
        return True
    
    def find_position(occupied, w, h, d):
        """Find the best (lowest z, then y, then x) position for a box."""
        best_pos = None
        best_key = None
        
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
                if best_pos is not None:
                    # Keep looking for better y,x at same z
                    pass
        
        return best_pos
    
    def greedy_pack(order, occupied_lists, placements_by_bin):
        """Place boxes in given order into bins using first-fit."""
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            for b in range(len(occupied_lists)):
                pos = find_position(occupied_lists[b], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    occupied_lists[b].append((x, y, z, w, h, d))
                    placements_by_bin[idx] = (b, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                b = len(occupied_lists)
                occupied_lists.append([])
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
        last_bin = max(bins_used)
        
        last_bin_vol = 0
        for idx in range(n):
            b, x, y, z, w, h, d = placements_by_bin[idx]
            if b == last_bin:
                last_bin_vol += w * h * d
        
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - occupied_frac)
        return score
    
    def simulated_annealing(placements_by_bin, occupied_lists, time_budget):
        """Simulated annealing to escape local minima via move and swap operations."""
        current_score = compute_score(placements_by_bin)
        best_score = current_score
        best_placements = dict(placements_by_bin)
        best_occupied = [list(b) for b in occupied_lists]
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        max_iters = 500
        
        start_time = time.time()
        
        for iteration in range(max_iters):
            if time.time() > start_time + time_budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Cool temperature
            progress = iteration / max_iters
            T = T_start * (T_end / T_start) ** progress
            
            # Choose random move type: 70% single box move, 30% swap
            move_type = random.random()
            
            if move_type < 0.7:
                # Single box move
                idx = random.randint(0, n - 1)
                b, x, y, z, w, h, d = placements_by_bin[idx]
                
                # Remove from current bin
                occupied_lists[b].remove((x, y, z, w, h, d))
                
                # Find all valid positions in all bins
                valid_moves = []
                for target_b in range(len(occupied_lists)):
                    pos = find_position(occupied_lists[target_b], w, h, d)
                    if pos is not None:
                        valid_moves.append((target_b, pos))
                
                if not valid_moves:
                    # Can't place anywhere, revert
                    occupied_lists[b].append((x, y, z, w, h, d))
                    continue
                
                # Pick a random valid move
                target_b, (nx, ny, nz) = random.choice(valid_moves)
                occupied_lists[target_b].append((nx, ny, nz, w, h, d))
                placements_by_bin[idx] = (target_b, nx, ny, nz, w, h, d)
                
                new_score = compute_score(placements_by_bin)
                delta = new_score - current_score
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_placements = dict(placements_by_bin)
                        best_occupied = [list(b) for b in occupied_lists]
                else:
                    # Revert
                    occupied_lists[target_b].remove((nx, ny, nz, w, h, d))
                    occupied_lists[b].append((x, y, z, w, h, d))
                    placements_by_bin[idx] = (b, x, y, z, w, h, d)
            
            else:
                # Swap two boxes
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                if i1 == i2:
                    continue
                
                b1, x1, y1, z1, w1, h1, d1 = placements_by_bin[i1]
                b2, x2, y2, z2, w2, h2, d2 = placements_by_bin[i2]
                
                # Remove both boxes
                occupied_lists[b1].remove((x1, y1, z1, w1, h1, d1))
                occupied_lists[b2].remove((x2, y2, z2, w2, h2, d2))
                
                # Try to place box1 in all bins and box2 in all bins
                pos1_candidates = []
                pos2_candidates = []
                
                for target_b in range(len(occupied_lists)):
                    p1 = find_position(occupied_lists[target_b], w1, h1, d1)
                    if p1 is not None:
                        pos1_candidates.append((target_b, p1))
                
                for target_b in range(len(occupied_lists)):
                    p2 = find_position(occupied_lists[target_b], w2, h2, d2)
                    if p2 is not None:
                        pos2_candidates.append((target_b, p2))
                
                if not pos1_candidates or not pos2_candidates:
                    # Revert
                    occupied_lists[b1].append((x1, y1, z1, w1, h1, d1))
                    occupied_lists[b2].append((x2, y2, z2, w2, h2, d2))
                    continue
                
                # Pick random valid positions
                tb1, (nx1, ny1, nz1) = random.choice(pos1_candidates)
                tb2, (nx2, ny2, nz2) = random.choice(pos2_candidates)
                
                # Check if placing both works (they might overlap if in same bin)
                occupied_lists[tb1].append((nx1, ny1, nz1, w1, h1, d1))
                occupied_lists[tb2].append((nx2, ny2, nz2, w2, h2, d2))
                
                # Verify no overlap if same bin
                if tb1 == tb2:
                    # Check for overlap between the two newly placed boxes
                    overlap = False
                    for (ox, oy, oz, ow, oh, od) in occupied_lists[tb1]:
                        if (ox, oy, oz, ow, oh, od) == (nx1, ny1, nz1, w1, h1, d1):
                            continue
                        if nx1 < ox + ow and ox < nx1 + w1 and \
                           ny1 < oy + oh and oy < ny1 + h1 and \
                           nz1 < oz + od and oz < nz1 + d1:
                            overlap = True
                            break
                    if overlap:
                        occupied_lists[tb1].remove((nx1, ny1, nz1, w1, h1, d1))
                        occupied_lists[tb2].remove((nx2, ny2, nz2, w2, h2, d2))
                        occupied_lists[b1].append((x1, y1, z1, w1, h1, d1))
                        occupied_lists[b2].append((x2, y2, z2, w2, h2, d2))
                        continue
                
                placements_by_bin[i1] = (tb1, nx1, ny1, nz1, w1, h1, d1)
                placements_by_bin[i2] = (tb2, nx2, ny2, nz2, w2, h2, d2)
                
                new_score = compute_score(placements_by_bin)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_placements = dict(placements_by_bin)
                        best_occupied = [list(b) for b in occupied_lists]
                else:
                    # Revert
                    occupied_lists[tb1].remove((nx1, ny1, nz1, w1, h1, d1))
                    occupied_lists[tb2].remove((nx2, ny2, nz2, w2, h2, d2))
                    occupied_lists[b1].append((x1, y1, z1, w1, h1, d1))
                    occupied_lists[b2].append((x2, y2, z2, w2, h2, d2))
                    placements_by_bin[i1] = (b1, x1, y1, z1, w1, h1, d1)
                    placements_by_bin[i2] = (b2, x2, y2, z2, w2, h2, d2)
        
        # Restore best solution
        for idx in range(n):
            placements_by_bin[idx] = best_placements[idx]
        for i in range(len(occupied_lists)):
            occupied_lists[i] = best_occupied[i]
        
        return best_score
    
    best_score = float('inf')
    best_placements = None
    
    for strat_name, order in strategies:
        if time.time() > deadline - 25:
            break
        
        # Multiple random restarts for each strategy
        n_restarts = 4
        for r in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Create a perturbed order for diversity
            if r == 0:
                order_copy = order[:]
            else:
                order_copy = order[:]
                # Apply random swaps to create diversity
                n_swaps = min(8, n // 2)
                for _ in range(n_swaps):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    order_copy[i], order_copy[j] = order_copy[j], order_copy[i]
            
            occupied_lists = []
            placements_by_bin = {}
            
            greedy_pack(order_copy, occupied_lists, placements_by_bin)
            
            if not placements_by_bin:
                continue
            
            # Use remaining time budget for simulated annealing
            time_remaining = deadline - time.time() - 5
            if time_remaining < 1:
                score = compute_score(placements_by_bin)
            else:
                score = simulated_annealing(placements_by_bin, occupied_lists, time_remaining)
            
            if score < best_score:
                best_score = score
                best_placements = dict(placements_by_bin)
    
    # Build final result
    if best_placements is None:
        # Fallback: simple shelf packing
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
