"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Sort indices by volume descending (FFD)
    order = sorted(range(n), key=lambda i: -box_vols[i])
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_spot(bin_placements, w, h, d):
        """Find a valid spot for a box in the bin. 
        Prioritizes positions with minimal x, then y, then z to pack tightly.
        Candidates are generated from the 'faces' of existing boxes.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidates from existing boxes
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Right face
            if px + pw + w <= W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
            # Top face
            if py + ph + h <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
            # Front face
            if pz + pd + d <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
        
        # Sort by "bottom-left-front" preference
        # We want to minimize z, then y, then x to fill from the corner
        sorted_candidates = sorted(candidates, key=lambda c: (c[2], c[1], c[0]))
        
        for (cx, cy, cz) in sorted_candidates:
            if can_place(bin_placements, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None

    best_score = float('inf')
    best_placement_info = None
    
    max_restarts = 100
    restart = 0
    
    while time.time() < deadline and restart < max_restarts:
        restart += 1
        
        # Create a perturbed ordering
        if restart == 1:
            cur_order = order[:]
        else:
            cur_order = order[:]
            # Perturb: random swaps
            for _ in range(min(10, n // 4)):
                i = random.randint(0, n - 2)
                cur_order[i], cur_order[i + 1] = cur_order[i + 1], cur_order[i]
        
        bins = []  # list of lists of (x, y, z, w, h, d)
        placement_info = {}  # box_index -> (bin_id, x, y, z)
        
        for idx in cur_order:
            w, h, d = boxes[idx]
            placed = False
            
            # Best Fit: Try all existing bins and pick the one that leaves the least space?
            # For simplicity and speed, we use First Fit with the improved spot finding.
            # However, Best Fit Decreasing often yields better bin counts.
            # Let's try to find the "best" bin by checking if it fits, and if so, 
            # we could potentially score the placement. But that's expensive.
            # Sticking to First Fit with good spot selection is usually fast and decent.
            
            for bin_id, bin_placements in enumerate(bins):
                spot = find_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    bin_placements.append((x, y, z, w, h, d))
                    placement_info[idx] = (bin_id, x, y, z)
                    placed = True
                    break
            
            if not placed:
                new_bin_id = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placement_info[idx] = (new_bin_id, 0, 0, 0)
        
        # Evaluate
        if not placement_info:
            continue
            
        occupied = set(info[0] for info in placement_info.values())
        B = len(occupied)
        last_bin = max(occupied)
        last_bin_vol = sum(box_vols[i] for i, info in placement_info.items() if info[0] == last_bin)
        last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - last_bin_frac)
        
        if score < best_score:
            best_score = score
            best_placement_info = dict(placement_info)
        
        if best_score <= 1.0 + 1e-9:
            break
    
    if best_placement_info is None:
        # Fallback to shelf if something went wrong
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
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
    
    # Build final result
    result = []
    for i in range(n):
        bin_id, x, y, z = best_placement_info[i]
        w, h, d = boxes[i]
        result.append((bin_id, x, y, z, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
