"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # We will use a list of bins. Each bin has a 3D numpy boolean array for occupancy.
    # bins[i] = (occupied_array, list_of_placements_in_bin)
    # But to keep memory reasonable, we only store the occupancy array and the count/volume.
    
    # For efficiency, we'll maintain a list of (occupied_3d, volume_used) for each bin.
    # occupied_3d is a np.zeros((W, H, D), dtype=bool)
    
    # To find the best position in a bin, we need to check all possible (x, y, z) positions.
    # A position (x, y, z) is feasible if the box (w, h, d) fits without exceeding W, H, D
    # and does not overlap with any occupied cell.
    # 
    # The directive suggests using np.any on a slice. However, checking ALL positions for EVERY
    # box in EVERY bin might be too slow if done naively (O(W*H*D) per box per bin).
    # 40 boxes, 5 instances. W,H,D are likely small enough (e.g., < 50).
    # Let's estimate: if W=H=D=20, then 8000 cells. 40 boxes * 8000 checks * 5 instances = 1.6M checks.
    # Each check involves a slice operation. This should be fast enough in NumPy.
    #
    # Strategy:
    # 1. Sort boxes? The problem says "preserving its original ordered dimensions (no rotations)".
    #    It does NOT say we must place them in the given order. We can reorder the input list
    #    to improve packing (e.g., largest volume first). The output must be a placement per box
    #    in the original order? No, "one placement (bin, x, y, z, w, h, d) per box".
    #    The return is a list of tuples. The evaluator likely maps the i-th tuple to the i-th box
    #    in the input `boxes`. So we must return the placements in the same order as the input `boxes`.
    #    We can internally sort the indices to process them in a better order, then unsort the results.
    
    # Let's try a "Best Fit" strategy:
    # - Process boxes in descending order of volume (or some heuristic).
    # - For each box, try to place it in an existing bin.
    #   - For each existing bin, find the "best" empty position.
    #     - "Best" could be defined as the position that minimizes the sum of distances to the
    #       occupied cells, or simply the first empty position found in a specific scan order
    #       (e.g., lowest x, then lowest y, then lowest z) to keep things compact.
    #     - To make it faster, we can use a "shelf" or "corner" heuristic but the directive
    #       asks for grid-based occupancy map and exhaustive evaluation of grid-aligned positions.
    #     - Let's define "best" as the position with the minimum (x + y + z) that is feasible.
    #       This tends to pack boxes towards the origin (0,0,0), which is good for leaving
    #       contiguous empty space for future boxes.
    #   - If no existing bin can fit the box, open a new bin.
    
    # Pre-compute box volumes
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Sort indices by descending volume
    indices = sorted(range(len(boxes)), key=lambda i: box_volumes[i], reverse=True)
    
    # bins: list of dicts or tuples
    # Each bin: { 'occupied': np.ndarray, 'vol': int, 'placements': [] }
    # We don't actually need to store placements per bin for the logic, just the occupied array.
    # But we need to track which box went where to return the final list.
    
    bin_data = []  # List of (occupied_array, volume_used)
    # result[i] = (bin_index, x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    def find_best_position(occupied, w, h, d):
        """
        Find the best position (x, y, z) in the given occupied grid for a box of size w, h, d.
        Returns (x, y, z) or None if no position found.
        'Best' is defined as the position with minimum x+y+z that is feasible.
        """
        # Pre-compute the maximum x, y, z
        max_x = W - w
        max_y = H - h
        max_z = D - d
        
        if max_x < 0 or max_y < 0 or max_z < 0:
            return None
            
        best_pos = None
        best_score = float('inf')
        
        # To optimize, we can iterate in an order that likely finds good spots early.
        # But to guarantee "minimum x+y+z", we might need to check all or use a smarter search.
        # Given small dimensions, checking all is fine.
        # However, we can prune: if x+y+z >= best_score, we can skip.
        
        # Let's iterate x, then y, then z.
        for x in range(max_x + 1):
            if x >= best_score: # Since y,z >= 0, x+y+z >= x
                break
            for y in range(max_y + 1):
                if x + y >= best_score:
                    break
                for z in range(max_z + 1):
                    score = x + y + z
                    if score >= best_score:
                        break
                    # Check if the region is free
                    # occupied[x:x+w, y:y+h, z:z+d]
                    if not np.any(occupied[x:x+w, y:y+h, z:z+d]):
                        best_pos = (x, y, z)
                        best_score = score
                        # We can't break immediately because there might be a better score?
                        # No, we are iterating in increasing order of x, then y, then z.
                        # The first valid position found in this nested loop order is NOT necessarily
                        # the one with minimum x+y+z.
                        # Example: (1,0,0) has score 1. (0,1,0) has score 1. (0,0,1) has score 1.
                        # (0,0,0) has score 0.
                        # If we find (1,0,0) first, best_score becomes 1.
                        # Then we check (0,1,0). x=0, y=1. score=1. 1 >= 1, so we break inner loop?
                        # Wait, the break condition `if score >= best_score: break` only breaks the innermost loop.
                        # We need to be careful.
                        
                        # Actually, if we want strictly minimum x+y+z, we should just find the minimum.
                        # Let's just collect the minimum.
                        pass
        
        # The above loop is a bit messy for optimization. Let's just do a clean scan for minimum score.
        # Since dimensions are small, O(W*H*D) is acceptable.
        
        best_pos = None
        min_score = float('inf')
        
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                for z in range(max_z + 1):
                    s = x + y + z
                    if s >= min_score:
                        continue
                    if not np.any(occupied[x:x+w, y:y+h, z:z+d]):
                        min_score = s
                        best_pos = (x, y, z)
                        
        return best_pos

    # To speed up, we can pre-calculate a "distance transform" or similar, but let's stick to the simple version first.
    # If it's too slow, we can optimize.
    
    for idx in indices:
        w, h, d = boxes[idx]
        
        placed = False
        best_bin_idx = -1
        best_pos_in_bin = None
        best_global_score = float('inf')
        
        # Try existing bins
        for b_idx in range(len(bin_data)):
            occupied, vol = bin_data[b_idx]
            pos = find_best_position(occupied, w, h, d)
            if pos is not None:
                # We prefer bins that are already "fuller" or where the position is "better"?
                # The score metric is: mean B + 0.1*(1 - last-bin occupied volume fraction).
                # Minimizing B is the primary goal. So we should prefer placing in an existing bin.
                # Among existing bins, does it matter which one?
                # If we place in bin A vs bin B, B stays the same.
                # The second term depends on the LAST bin (largest label).
                # If we add to an existing bin, the last bin might not change (unless we open a new one).
                # If we open a new bin, B increases by 1.
                # So, always prefer existing bins over new bins.
                # If multiple existing bins can fit, which one to choose?
                # It might affect future placements. A heuristic: choose the bin where the box fits
                # in the "tightest" spot (lowest x+y+z) or the bin with the most volume used?
                # Let's just pick the first one that fits with the lowest x+y+z in that bin?
                # Or simply the first bin that fits?
                # Let's try to pick the bin that minimizes the resulting "wasted space" or something.
                # For simplicity, let's just pick the first bin that can accommodate the box.
                # To improve, we can pick the bin where the box is placed at the lowest x+y+z.
                
                s = pos[0] + pos[1] + pos[2]
                if s < best_global_score:
                    best_global_score = s
                    best_bin_idx = b_idx
                    best_pos_in_bin = pos
        
        if best_bin_idx != -1:
            # Place in best_bin_idx
            x, y, z = best_pos_in_bin
            occupied, vol = bin_data[best_bin_idx]
            occupied[x:x+w, y:y+h, z:z+d] = True
            vol += w * h * d
            bin_data[best_bin_idx] = (occupied, vol)
            result[idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            new_bin_idx = len(bin_data)
            occupied = np.zeros((W, H, D), dtype=bool)
            occupied[0:w, 0:h, 0:d] = True
            vol = w * h * d
            bin_data.append((occupied, vol))
            result[idx] = (new_bin_idx, 0, 0, 0, w, h, d)
            
    return result

# EVOLVE-BLOCK-END
