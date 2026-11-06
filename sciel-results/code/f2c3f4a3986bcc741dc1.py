"""3D Container Loading with multi-strategy shelf packing, simulated annealing, and gap-filling."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 118.0  # safety margin before 120s kill
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    def shelf_pack_3d(ordered_indices):
        """
        3D shelf packing with improved position search:
        - Y-axis is the stacking axis (shelves at different y levels)
        - Each shelf has a fixed height
        - Within a shelf, boxes are placed in the X-Z plane
        - Uses candidate positions from existing boxes plus 0
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        shelves = []
        current_y = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            placed = False
            for shelf in shelves:
                if shelf['y'] + shelf['h'] > H:
                    continue
                for ow, oh, od in orients:
                    if oh > shelf['h']:
                        continue
                    if ow > W or od > D:
                        continue
                    
                    # Generate candidate x positions
                    x_candidates = {0}
                    for px, pz, pow_, pod, poh in shelf['boxes']:
                        if px + pow_ <= W - ow:
                            x_candidates.add(px + pow_)
                    x_candidates = sorted(x_candidates)
                    
                    found_pos = False
                    for x in x_candidates:
                        if x + ow > W:
                            continue
                        z_candidates = {0}
                        for px, pz, pow_, pod, poh in shelf['boxes']:
                            if pz + pod <= D - od:
                                z_candidates.add(pz + pod)
                        z_candidates = sorted(z_candidates)
                        
                        for z in z_candidates:
                            if z + od > D:
                                continue
                            free = True
                            for px, pz, pow_, pod, poh in shelf['boxes']:
                                if not (x + ow <= px or px + pow_ <= x):
                                    if not (z + od <= pz or pz + pod <= z):
                                        free = False
                                        break
                            if free:
                                placements[idx] = (0, x, int(shelf['y']), z, ow, oh, od)
                                shelf['boxes'].append((x, z, ow, od, oh))
                                total_volume += ow * oh * od
                                placed = True
                                found_pos = True
                                break
                        if found_pos:
                            break
                    if placed:
                        break
                if placed:
                    break
            
            if placed:
                continue
            
            # Try to start a new shelf
            best_orient = None
            for ow, oh, od in orients:
                if current_y + oh <= H and ow <= W and od <= D:
                    if best_orient is None or oh < best_orient[1]:
                        best_orient = (ow, oh, od)
            
            if best_orient:
                ow, oh, od = best_orient
                new_shelf = {'y': current_y, 'h': oh, 'boxes': [(0, 0, ow, od, oh)]}
                shelves.append(new_shelf)
                current_y += oh
                placements[idx] = (0, 0, int(current_y - oh), 0, ow, oh, od)
                total_volume += ow * oh * od
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
        lambda i: (boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(1, min(boxes[i][0], boxes[i][1], boxes[i][2])),  # Volume/min
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]) / max(1, min(boxes[i][0], boxes[i][1], boxes[i][2])),  # Aspect ratio
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    # Run each static sorting strategy
    for sort_key in strategies:
        if time.time() > deadline - 20:
            break
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = shelf_pack_3d(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v
    
    # Simulated annealing to find better orderings
    if time.time() < deadline - 30:
        # Start from the best static ordering
        best_ordered = sorted(indexed_boxes, key=strategies[0], reverse=True)
        current_volume = best_volume
        current_placements = best_placements
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        max_iter = 2000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed ordering by swapping two adjacent elements
            new_ordered = current_ordered[:]
            if len(new_ordered) > 1:
                pos = random.randint(0, len(new_ordered) - 2)
                new_ordered[pos], new_ordered[pos + 1] = new_ordered[pos + 1], new_ordered[pos]
            
            p, v = shelf_pack_3d(new_ordered)
            
            delta_v = v - current_volume
            if delta_v >= 0 or random.random() < np.exp(delta_v / (T * max(1, W * H * D / 1000))):
                current_ordered = new_ordered
                current_volume = v
                current_placements = p
                if v > best_volume:
                    best_volume = v
                    best_placements = p
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_placements
# EVOLVE-BLOCK-END
