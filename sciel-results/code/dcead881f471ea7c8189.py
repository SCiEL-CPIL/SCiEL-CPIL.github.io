"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Generate all 6 orientations for a box (unique)
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique
    
    # Pre-compute orientations
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
    
    # Volume and other features
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Sort indices by various criteria
    order_vol_desc = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    order_max_dim_desc = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    order_aspect_asc = sorted(range(n), key=lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i]) > 0 else float('inf'))
    
    # Helper to check if a box fits in the grid at (x, y, z) with dims (bw, bh, bd)
    # Using NumPy slicing for efficiency
    def can_place(grid, x, y, z, bw, bh, bd):
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check if any part of the target region is occupied (True)
        # grid[x:x+bw, y:y+bh, z:z+bd]
        if np.any(grid[x:x+bw, y:y+bh, z:z+bd]):
            return False
        return True

    def find_position(grid, bw, bh, bd):
        """
        Find the best position for a box.
        Strategy: Scan from bottom (z=0) up, then back (y=0) to front, then left (x=0) to right.
        This is a standard 'Bottom-Left-Back' heuristic which tends to produce stable stacks.
        """
        # Iterate z from 0 to D-bd
        # To speed up, we can check entire slices if possible, but simple loop is often fast enough for small grids
        # However, with 60 boxes, grid size might be moderate.
        # Let's optimize the scan order.
        
        # Pre-check: if box doesn't fit in empty space, return None
        # We will scan z levels.
        
        best_pos = None
        # We want the "lowest" position.
        # The coordinate system in the output is (x, y, z).
        # Let's assume z is height (vertical), y is depth, x is width.
        # The prompt says (bin, x, y, z, w, h, d).
        # Usually in 3D packing:
        # x: width (0..W)
        # y: height (0..H) OR depth?
        # The container is (W, H, D).
        # The box is (w, h, d).
        # The placement is (x, y, z).
        # The dimensions placed are (w, h, d).
        # So x corresponds to W, y to H, z to D?
        # Let's check the can_place logic in the seed:
        # if x + bw > W or y + bh > H or z + bd > D
        # So x is along W, y is along H, z is along D.
        
        # "Bottom" usually means lowest Z (if Z is up) or lowest Y (if Y is up).
        # In the seed's score: cz * 10000 + cy * 100 + cx.
        # This implies Z is the most significant (bottom-most), then Y, then X.
        # So we should prioritize low Z, then low Y, then low X.
        
        # Scan Z from 0 to D-bd
        for z in range(D - bd + 1):
            # Scan Y from 0 to H-bh
            for y in range(H - bh + 1):
                # Scan X from 0 to W-bw
                for x in range(W - bw + 1):
                    if can_place(grid, x, y, z, bw, bh, bd):
                        return (x, y, z)
        return None

    def run_pack(order, orientations_override=None):
        """Try to pack boxes in the given order."""
        # Initialize grid
        # Dimensions: W x H x D
        grid = np.zeros((W, H, D), dtype=np.uint8)
        placed = []
        results = [None] * n
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = orientations_override[idx] if orientations_override else box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            
            # Try all orientations
            # Sort orientations by "stability" or just try all
            for (ow, oh, od) in orientations:
                pos = find_position(grid, ow, oh, od)
                if pos is not None:
                    # Since find_position returns the first valid in BLB order, 
                    # the first orientation that fits is "best" in terms of BLB priority.
                    # However, different orientations might fit in different places.
                    # The BLB scan finds the absolute lowest position for a specific orientation.
                    # We should compare the positions found for different orientations.
                    # But find_position is expensive.
                    # Optimization: If we find a position at (0,0,0), it's the best possible.
                    cx, cy, cz = pos
                    if best_pos is None:
                        best_pos = pos
                        best_orientation = (ow, oh, od)
                    else:
                        # Compare (cx, cy, cz) with best_pos
                        # We want min z, then min y, then min x
                        if cz < best_pos[2] or (cz == best_pos[2] and cy < best_pos[1]) or (cz == best_pos[2] and cy == best_pos[1] and cx < best_pos[0]):
                            best_pos = pos
                            best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                # Mark grid
                grid[cx:cx+ow, cy:cy+oh, cz:cz+od] = 1
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results

    def compute_volume(results):
        return sum(w * h * d for (b, x, y, z, w, h, d) in results if b == 0)
    
    # Try multiple packing orders
    best_results = None
    best_volume = -1
    
    # Order 1: Volume descending
    results1 = run_pack(order_vol_desc)
    vol1 = compute_volume(results1)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = results1
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 2: Max dimension descending
    results2 = run_pack(order_max_dim_desc)
    vol2 = compute_volume(results2)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = results2
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 3: Aspect ratio ascending (cubic first)
    results3 = run_pack(order_aspect_asc)
    vol3 = compute_volume(results3)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = results3
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try random orders
    random.seed(42)
    for trial in range(300):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order)
        vol = compute_volume(results)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    # Try perturbed volume-descending orders
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        for trial in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            order = list(order_vol_desc)
            for _ in range(random.randint(1, 5)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                order[i], order[j] = order[j], order[i]
            
            results = run_pack(order)
            vol = compute_volume(results)
            if vol > best_volume:
                best_volume = vol
                best_results = results
    
    # Final reinsertion pass on best result
    # The grid approach makes reinsertion tricky because we don't have the grid state of the best result easily accessible
    # unless we re-run or store it.
    # Since run_pack creates a local grid, we would need to modify run_pack to return the grid or re-run.
    # Given the speed of the grid method, re-running is cheap.
    # But "reinsertion" usually implies trying to fit left-out boxes into the existing packing.
    # With the grid method, if a box was left out, it means it didn't fit in the BLB scan.
    # It might fit if we rearranged others, but that's a different search.
    # The previous code had a try_reinsert that just tried to find a spot for left-out boxes in the *existing* placement.
    # With the grid, we can reconstruct the grid from best_results and try to insert.
    
    if best_results is not None and time.time() - start_time < DEADLINE - 5:
        left_out = [i for i in range(n) if best_results[i][0] == -1]
        if left_out:
            # Reconstruct grid
            grid = np.zeros((W, H, D), dtype=np.uint8)
            for (b, x, y, z, w, h, d) in best_results:
                if b == 0:
                    grid[x:x+w, y:y+h, z:z+d] = 1
            
            # Try to insert left-out boxes
            # We need to be careful not to overlap.
            # We can iterate through left_out boxes and try to place them.
            # If placed, update grid and results.
            for li_idx in left_out:
                bw, bh, bd = boxes[li_idx]
                orientations = box_orientations[li_idx]
                
                best_pos = None
                best_orientation = None
                
                for (ow, oh, od) in orientations:
                    pos = find_position(grid, ow, oh, od)
                    if pos is not None:
                        cx, cy, cz = pos
                        if best_pos is None:
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                        else:
                            if cz < best_pos[2] or (cz == best_pos[2] and cy < best_pos[1]) or (cz == best_pos[2] and cy == best_pos[1] and cx < best_pos[0]):
                                best_pos = pos
                                best_orientation = (ow, oh, od)
                
                if best_pos is not None:
                    cx, cy, cz = best_pos
                    ow, oh, od = best_orientation
                    grid[cx:cx+ow, cy:cy+oh, cz:cz+od] = 1
                    best_results[li_idx] = (0, cx, cy, cz, ow, oh, od)
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
