"""
Strategy: Multi-Objective Local Search (Simulated Annealing) on Box Ordering.
Unlike the static greedy approach, this method iteratively perturbs the box
placement order and re-evaluates the packing using a 'Best-Fit' heuristic that
minimizes the "waste" (distance from origin) of the chosen position. This allows
the algorithm to escape the local optima of simple volume/size sorting by
exploring non-intuitive sequences that might allow larger boxes to fit in gaps
created by specific ordering of smaller boxes.
"""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin for 120s limit
    
    # Precompute orientations for each box
    # Unique permutations of (w, h, d)
    all_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))
        
    # Precompute volumes
    volumes = [w*h*d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Helper: Check if a box fits at (x, y, z) with dims (bw, bh, bd)
    def fits(x, y, z, bw, bh, bd, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or
                    y + bh <= py or py + ph <= y or
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    # Helper: Generate candidate anchor points
    def get_anchors(placed):
        anchors = set()
        anchors.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            # Faces
            anchors.add((px + pw, py, pz))
            anchors.add((px, py + ph, pz))
            anchors.add((px, py, pz + pd))
            # Edges/Corners
            anchors.add((px + pw, py + ph, pz))
            anchors.add((px + pw, py, pz + pd))
            anchors.add((px, py + ph, pz + pd))
            anchors.add((px + pw, py + ph, pz + pd))
        
        valid_anchors = []
        for a in anchors:
            if 0 <= a[0] <= W and 0 <= a[1] <= H and 0 <= a[2] <= D:
                valid_anchors.append(a)
        return valid_anchors

    # Helper: Pack a specific order of boxes using Best-Fit heuristic
    # Returns (filled_volume, result_list)
    def pack_order(order):
        placed = []
        result = [None] * n
        filled = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_cost = float('inf')
            
            # Get anchors once per box to save time
            anchors = get_anchors(placed)
            
            # Try all orientations
            for bw, bh, bd in all_orients[idx]:
                # If box is too big for container in this orientation, skip
                if bw > W or bh > H or bd > D:
                    continue
                    
                for ax, ay, az in anchors:
                    if fits(ax, ay, az, bw, bh, bd, placed):
                        # Best-Fit heuristic: Minimize "waste"
                        # We want to place boxes as close to (0,0,0) as possible
                        # to keep the "frontier" compact
                        cost = ax + ay + az
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (ax, ay, az)
                            best_orient = (bw, bh, bd)
                            
                # Optimization: If we found a spot at (0,0,0) or very low cost, 
                # we might break early, but checking all anchors ensures best fit.
                # However, for speed, we can break if cost is 0.
                if best_cost == 0:
                    break

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # Initial Order: Volume Descending (standard heuristic)
    initial_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    best_filled, best_result = pack_order(initial_order)
    
    # Simulated Annealing on the ordering
    # We swap two boxes in the order and see if it improves the packing
    current_order = initial_order[:]
    current_filled, current_result = pack_order(current_order)
    
    T = 1000.0  # Initial temperature
    cooling = 0.995
    min_T = 1.0
    
    # Time budget for SA loop
    sa_start = time.time()
    
    while T > min_T and time.time() < deadline - 10.0:
        # Pick two random indices to swap
        i, j = random.sample(range(n), 2)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_filled, new_result = pack_order(current_order)
        delta = new_filled - current_filled
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_filled = new_filled
            current_result = new_result
            if current_filled > best_filled:
                best_filled = current_filled
                best_result = current_result
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        T *= cooling
        
        # Check time
        if time.time() > deadline - 10.0:
            break

    # If we still have time, try a few more static heuristics that might have been missed
    # or just return the best found.
    
    # Fallback if something went wrong (shouldn't happen)
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_result

# EVOLVE-BLOCK-END
