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
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    
    # Sort indices by volume descending (FFD)
    order = sorted(range(n), key=lambda i: -box_vols[i])
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap in all three dimensions
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        return True
    
    def find_spot(bin_placements, w, h, d):
        """Find a valid spot for a box in the bin. Try to minimize waste."""
        # Collect candidate positions
        # Strategy: try positions at origin, and along edges of existing boxes
        candidates = [(0, 0, 0)]
        
        # Add positions based on existing placements
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Right of existing box
            if px + pw + w <= W:
                candidates.append((px + pw, py, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px + pw, py + ph, pz))
            # Above existing box
            if py + ph + h <= H:
                candidates.append((px, py + ph, pz))
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px, py + ph, pz + pd))
            # In front of existing box
            if pz + pd + d <= D:
                candidates.append((px, py, pz + pd))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
        
        # Also try some random positions for diversity
        for _ in range(5):
            rx = random.randint(0, max(0, W - w))
            ry = random.randint(0, max(0, H - h))
            rz = random.randint(0, max(0, D - d))
            candidates.append((rx, ry, rz))
        
        # Sort candidates by "bottom-left-front" preference to pack tightly
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        for (cx, cy, cz) in candidates:
            if can_place(bin_placements, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None
    
    def bin_volume(bin_placements, boxes_in_bin):
        """Calculate total volume of boxes in a bin."""
        return sum(box_vols[i] for i in boxes_in_bin)
    
    def evaluate(placement_map, bins):
        """Evaluate the current packing: B + 0.1*(1 - last_bin_frac)"""
        if not bins:
            return float('inf')
        
        # Count occupied bins
        occupied = set(placement_map.values())
        B = len(occupied)
        
        if B == 0:
            return float('inf')
        
        # Find the last bin (largest label)
        last_bin = max(occupied)
        
        # Calculate volume in last bin
        last_bin_vol = 0
        for i, b in placement_map.items():
            if b == last_bin:
                last_bin_vol += box_vols[i]
        
        last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        return B + 0.1 * (1 - last_bin_frac)
    
    # FFD placement
    best_result = None
    best_score = float('inf')
    
    # Try multiple random restarts with different orderings
    max_restarts = 50
    restart = 0
    
    while time.time() < deadline and restart < max_restarts:
        restart += 1
        
        # Create a perturbed ordering
        if restart == 1:
            cur_order = order[:]
        else:
            cur_order = order[:]
            # Perturb: swap some adjacent elements
            for _ in range(min(10, n // 4)):
                i = random.randint(0, n - 2)
                cur_order[i], cur_order[i + 1] = cur_order[i + 1], cur_order[i]
        
        # FFD placement
        bins = []  # list of lists of (x, y, z, w, h, d)
        placement_map = {}  # box_index -> bin_id
        
        for idx in cur_order:
            w, h, d = boxes[idx]
            placed = False
            
            # Try to place in existing bins (first-fit)
            for bin_id, bin_placements in enumerate(bins):
                spot = find_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    bin_placements.append((x, y, z, w, h, d))
                    placement_map[idx] = bin_id
                    placed = True
                    break
            
            # If not placed, create new bin
            if not placed:
                new_bin_id = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placement_map[idx] = new_bin_id
        
        # Evaluate
        score = evaluate(placement_map, bins)
        
        if score < best_score:
            best_score = score
            best_result = dict(placement_map)
        
        # Early termination if we can't do better than 1 bin
        if best_score <= 1.0 + 1e-9:
            break
    
    # If no result found (shouldn't happen), use simple shelf
    if best_result is None:
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
    
    # Convert placement_map to the required format
    # We need to reconstruct (bin, x, y, z, w, h, d) for each box
    # Rebuild bins from best_result
    bins_dict = {}
    for idx, bin_id in best_result.items():
        if bin_id not in bins_dict:
            bins_dict[bin_id] = []
        bins_dict[bin_id].append(idx)
    
    # For each box, we need to find its position in the bin
    # We'll re-simulate the placement to get positions
    # Actually, we need to store positions. Let's redo this properly.
    
    # Let's re-implement to track positions
    # We'll run the best ordering again and track positions
    
    # Find the ordering that gave best_result
    # Actually, we lost track of which ordering gave the best result.
    # Let's modify the approach: store the full placement with positions.
    
    # Restart with position tracking
    best_result_with_pos = None
    best_score2 = float('inf')
    
    restart2 = 0
    max_restarts2 = 30
    
    while time.time() < deadline and restart2 < max_restarts2:
        restart2 += 1
        
        if restart2 == 1:
            cur_order = order[:]
        else:
            cur_order = order[:]
            for _ in range(min(10, n // 4)):
                i = random.randint(0, n - 2)
                cur_order[i], cur_order[i + 1] = cur_order[i + 1], cur_order[i]
        
        bins = []
        placement_info = {}  # box_index -> (bin_id, x, y, z)
        
        for idx in cur_order:
            w, h, d = boxes[idx]
            placed = False
            
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
        occupied = set(info[0] for info in placement_info.values())
        B = len(occupied)
        last_bin = max(occupied) if occupied else 0
        last_bin_vol = sum(box_vols[i] for i, info in placement_info.items() if info[0] == last_bin)
        last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        score2 = B + 0.1 * (1 - last_bin_frac)
        
        if score2 < best_score2:
            best_score2 = score2
            best_result_with_pos = dict(placement_info)
        
        if best_score2 <= 1.0 + 1e-9:
            break
    
    if best_result_with_pos is None:
        # Fallback to shelf
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
        bin_id, x, y, z = best_result_with_pos[i]
        w, h, d = boxes[i]
        result.append((bin_id, x, y, z, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
