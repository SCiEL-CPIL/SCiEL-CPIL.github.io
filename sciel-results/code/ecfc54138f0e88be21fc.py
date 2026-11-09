import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Generate all 6 orientations for a box
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
    
    # Sort boxes by volume descending (larger first)
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at position (x, y, z) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        return True
    
    def find_position(placed, bw, bh, bd):
        """Find the best position for a box using corner-based heuristic."""
        candidates = [(0, 0, 0)]
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py, pz))
        
        candidates = list(set(candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if not can_place(placed, cx, cy, cz, bw, bh, bd):
                continue
            
            score = cz * 100 + cy * 10 + cx
            
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    def run_pack(order):
        """Try to pack boxes in the given order."""
        placed = []
        results = [None] * n
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results
    
    def run_shelf_pack(order):
        """Shelf-based packing: place boxes in horizontal layers."""
        placed = []
        results = [None] * n
        shelf_y = 0  # Current shelf height (y-coordinate)
        shelf_x = 0  # Current x position on this shelf
        shelf_z = 0  # Current z position on this shelf
        shelf_max_h = 0  # Max height of boxes on current shelf
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                # Try placing on current shelf
                if shelf_x + ow <= W and shelf_z + od <= D and oh <= H - shelf_y:
                    if shelf_x == 0:
                        # New shelf, reset z
                        shelf_z = 0
                        shelf_max_h = oh
                    elif oh <= shelf_max_h:
                        # Fits on current shelf
                        pos = (shelf_x, shelf_y, shelf_z)
                        if can_place(placed, pos[0], pos[1], pos[2], ow, oh, od):
                            score = shelf_y * 100 + shelf_x * 10 + shelf_z
                            if score < best_score:
                                best_score = score
                                best_pos = pos
                                best_orientation = (ow, oh, od)
                
                # Also try general position search
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cy * 100 + cx * 10 + cz
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
                
                # Update shelf state if placed on current shelf
                if cy == shelf_y:
                    shelf_x += ow
                    if shelf_x >= W:
                        # Move to next shelf
                        shelf_x = 0
                        shelf_z = 0
                        shelf_y += shelf_max_h
                        shelf_max_h = 0
                    else:
                        shelf_z += od
                        if shelf_z >= D:
                            shelf_z = 0
                else:
                    # Placed elsewhere, reset shelf tracking
                    shelf_x = 0
                    shelf_z = 0
                    shelf_max_h = 0
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results
    
    def reinsert_leftout(results, placed):
        """Try to re-insert left-out boxes into gaps."""
        leftout = [i for i in range(n) if results[i][0] == -1]
        if not leftout:
            return results, placed
        
        # Sort left-out by volume descending
        leftout.sort(key=lambda i: box_volumes[i], reverse=True)
        
        for idx in leftout:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
        
        return results, placed
    
    # Try multiple packing orders
    best_results = None
    best_volume = -1
    
    # Order 1: Original order
    order1 = list(range(n))
    results1 = run_pack(order1)
    vol1 = sum(w*h*d for (b, x, y, z, w, h, d) in results1 if b == 0)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = results1
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 2: Sorted by volume descending
    order2 = box_indices
    results2 = run_pack(order2)
    vol2 = sum(w*h*d for (b, x, y, z, w, h, d) in results2 if b == 0)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = results2
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 3: Sorted by max dimension descending
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    order3 = sorted(range(n), key=lambda i: box_max_dim[i], reverse=True)
    results3 = run_pack(order3)
    vol3 = sum(w*h*d for (b, x, y, z, w, h, d) in results3 if b == 0)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = results3
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 4: Sorted by aspect ratio (more cubic first)
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else float('inf') for w, h, d in boxes]
    order4 = sorted(range(n), key=lambda i: box_aspect[i])
    results4 = run_pack(order4)
    vol4 = sum(w*h*d for (b, x, y, z, w, h, d) in results4 if b == 0)
    if vol4 > best_volume:
        best_volume = vol4
        best_results = results4
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try shelf-based packing
    results5 = run_shelf_pack(box_indices)
    vol5 = sum(w*h*d for (b, x, y, z, w, h, d) in results5 if b == 0)
    if vol5 > best_volume:
        best_volume = vol5
        best_results = results5
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try many random orders if time permits
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order)
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    # Try local improvement: perturbed orders based on best
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        base_orders = [box_indices, order3, order4]
        for base_order in base_orders:
            for trial in range(50):
                if time.time() - start_time > DEADLINE - 5:
                    break
                order = list(base_order)
                num_swaps = random.randint(1, 8)
                for _ in range(num_swaps):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    order[i], order[j] = order[j], order[i]
                
                results = run_pack(order)
                vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
                if vol > best_volume:
                    best_volume = vol
                    best_results = results
    
    # Reinsert left-out boxes
    if best_results is not None and time.time() - start_time < DEADLINE - 5:
        placed = [(x, y, z, w, h, d) for (b, x, y, z, w, h, d) in best_results if b == 0]
        best_results, _ = reinsert_leftout(best_results, placed)
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
