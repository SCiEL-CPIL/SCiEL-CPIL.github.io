"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import heapq

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Check if a box fits in a given position without overlapping existing placements
    def fits(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            # Check for overlap
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    # Generate candidate anchor positions for placing a new box
    def get_anchors(placed, W, H, D):
        anchors = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placed:
            # Try placing next to each existing box in 6 directions
            if px + pw <= W:
                anchors.append((px + pw, py, pz))
            if py + ph <= H:
                anchors.append((px, py + ph, pz))
            if pz + pd <= D:
                anchors.append((px, py, pz + pd))
            # Also try corners
            if px + pw <= W and py + ph <= H:
                anchors.append((px + pw, py + ph, pz))
            if px + pw <= W and pz + pd <= D:
                anchors.append((px + pw, py, pz + pd))
            if py + ph <= H and pz + pd <= D:
                anchors.append((px, py + ph, pz + pd))
        # Remove duplicates
        return list(set(anchors))
    
    # Try to pack boxes using a greedy approach with multiple sorting strategies
    def try_pack(box_indices, order_key):
        placed = []
        result = [None] * len(boxes)
        
        # Sort boxes according to the order key
        sorted_indices = sorted(box_indices, key=lambda i: order_key(boxes[i]), reverse=True)
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            best_pos = None
            best_vol = 0
            
            for (ow, oh, od) in orientations:
                if ow > W or oh > H or od > D:
                    continue
                anchors = get_anchors(placed, W, H, D)
                for (ax, ay, az) in anchors:
                    if fits(ax, ay, az, ow, oh, od, placed):
                        # Prefer positions closer to origin (lower x, y, z)
                        priority = ax * D * H + ay * D + az
                        if best_pos is None or priority < best_vol:
                            best_pos = (ax, ay, az, ow, oh, od)
                            best_vol = priority
            
            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return result
    
    # Define multiple sorting strategies
    def strategy_volume(box):
        return box[0] * box[1] * box[2]
    
    def strategy_max_dim(box):
        return max(box)
    
    def strategy_min_dim(box):
        return min(box)
    
    def strategy_surface_area(box):
        w, h, d = box
        return 2 * (w*h + h*d + d*w)
    
    def strategy_aspect_ratio(box):
        w, h, d = box
        max_d = max(w, h, d)
        min_d = min(w, h, d)
        return max_d / min_d if min_d > 0 else 0
    
    strategies = [
        ("volume", strategy_volume),
        ("max_dim", strategy_max_dim),
        ("min_dim", strategy_min_dim),
        ("surface_area", strategy_surface_area),
        ("aspect_ratio", strategy_aspect_ratio),
    ]
    
    box_indices = list(range(len(boxes)))
    best_result = None
    best_fill = 0
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
        
        result = try_pack(box_indices, key_func)
        
        # Calculate fill ratio
        filled_volume = sum(w*h*d for (_, x, y, z, w, h, d) in result if _ == 0)
        fill_ratio = filled_volume / (W * H * D) if W * H * D > 0 else 0
        
        if fill_ratio > best_fill:
            best_fill = fill_ratio
            best_result = result
    
    # If we have time, try a second pass with different ordering on remaining boxes
    if best_result is not None and time.time() < deadline - 20:
        # Try to re-pack boxes that were left out
        left_out = [i for i, p in enumerate(best_result) if p[0] == -1]
        if left_out:
            # Try each strategy on the left-out boxes only
            for name, key_func in strategies:
                if time.time() > deadline:
                    break
                
                # Create a modified problem with only left-out boxes
                # We'll try to fit them into the remaining space
                placed = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result if p[0] == 0]
                
                improved = False
                for idx in left_out:
                    w, h, d = boxes[idx]
                    orientations = get_orientations(w, h, d)
                    for (ow, oh, od) in orientations:
                        if ow > W or oh > H or od > D:
                            continue
                        anchors = get_anchors(placed, W, H, D)
                        for (ax, ay, az) in anchors:
                            if fits(ax, ay, az, ow, oh, od, placed):
                                placed.append((ax, ay, az, ow, oh, od))
                                best_result[idx] = (0, ax, ay, az, ow, oh, od)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    # Recalculate fill ratio
                    filled_volume = sum(w*h*d for (_, x, y, z, w, h, d) in best_result if _ == 0)
                    fill_ratio = filled_volume / (W * H * D) if W * H * D > 0 else 0
                    if fill_ratio > best_fill:
                        best_fill = fill_ratio
    
    if best_result is None:
        # Fallback: return all boxes as not placed
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
