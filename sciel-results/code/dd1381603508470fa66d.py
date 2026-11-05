"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (evaluator kills at 150s, keep 5s buffer)
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute unique orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orientations.append(orients)

    # Sort indices by volume descending for initial ordering
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    def run_packing(order, time_limit):
        """
        Runs a single packing instance using grid-based occupancy and local search.
        """
        # Use a 3D numpy array for occupancy. 
        # Resolution: 1 unit. If dimensions are large, this might be memory heavy, 
        # but for typical contest sizes (e.g., < 100x100x100) it's fine.
        # If dimensions are huge, we'd need a different structure, but let's assume reasonable bounds first.
        # To be safe against memory, cap grid size or fall back to list if too big.
        # For now, assume W, H, D are integers or cast to int.
        Wi, Hi, Di = int(W), int(H), int(D)
        
        # If dimensions are non-integer, we lose precision. 
        # The task implies continuous or integer? "w, h, d" usually integers in these tasks. 
        # Let's assume integer dimensions for the grid approach. 
        # If not integer, we scale? No, let's stick to integer assumption for the grid method's validity.
        # If inputs are floats, we might have issues. Let's check type.
        if not all(isinstance(x, int) for x in [W, H, D]):
            # Fallback to list-based if dimensions are not integers, or just cast?
            # Casting might cause overlaps if 0.5 is involved. 
            # Given the "score" nature, likely integer boxes.
            # Let's cast to int for grid, but this is risky. 
            # Alternative: Use the list-based method if non-int.
            # For now, let's assume integer dimensions as per standard bin packing.
            pass

        if Wi * Hi * Di > 10_000_000: # 10M cells is ~80MB, might be too slow/memory heavy for 120s?
             # Fallback to simpler list-based greedy if grid is too big
             return run_list_based_greedy(order, time_limit)

        occupancy = np.zeros((Wi, Hi, Di), dtype=np.uint8)
        placed_boxes = [] # List of (idx, x, y, z, w, h, d)
        results = [None] * len(boxes)
        
        # Initialize results to -1
        for i in range(len(boxes)):
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > Wi or y + h > Hi or z + d > Di:
                return False
            # Check occupancy
            if np.any(occupancy[x:x+w, y:y+h, z:z+d] > 0):
                return False
            return True

        def place(x, y, z, w, h, d, idx):
            occupancy[x:x+w, y:y+h, z:z+d] = 1
            placed_boxes.append((idx, x, y, z, w, h, d))
            results[idx] = (0, x, y, z, w, h, d)

        def remove(idx):
            # Find and remove
            for i, (bi, bx, by, bz, bw, bh, bd) in enumerate(placed_boxes):
                if bi == idx:
                    occupancy[bx:bx+bw, by:by+bh, bz:bz+bd] = 0
                    placed_boxes.pop(i)
                    results[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                    return
            # Should not happen

        # Phase 1: Greedy Placement
        # Sort order is provided.
        # We use "First Fit" in a specific traversal order: 
        # Minimize Z, then Y, then X (pack to bottom, back, left)
        # This is deterministic. To add diversity, we can randomize the traversal order slightly?
        # Or just rely on the randomized input order.
        
        for idx in order:
            if time.time() > time_limit:
                break
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            # Try to place. 
            # We iterate through all possible positions? That's O(W*H*D*Orients). 
            # Too slow. 
            # We need a smarter candidate generation.
            # "Lowest-Left-Front" usually implies scanning from (0,0,0) upwards.
            # We can scan Z from 0..Di, Y from 0..Hi, X from 0..Wi.
            # First valid spot is taken.
            
            placed = False
            # Optimization: Precompute max extents? No, dynamic.
            # Just scan.
            # To speed up, we can break early if we find a spot.
            
            # Sort orientations by volume? No, by "compactness" or just fixed.
            # Let's try orientations in fixed order.
            
            for ow, oh, od in orients:
                if placed:
                    break
                # Scan Z
                for z in range(Di - od + 1):
                    if placed:
                        break
                    # Scan Y
                    for y in range(Hi - oh + 1):
                        if placed:
                            break
                        # Scan X
                        for x in range(Wi - ow + 1):
                            if can_place(x, y, z, ow, oh, od):
                                place(x, y, z, ow, oh, od, idx)
                                placed = True
                                break
            if not placed:
                # Box left out for now
                pass

        # Phase 2: Local Search / Improvement
        # Try to place unplaced boxes by moving existing ones.
        # Identify unplaced boxes
        unplaced = [i for i in range(len(boxes)) if results[i][0] == -1]
        
        # Sort unplaced by volume descending (try to fit big ones first)
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        # We will try to fit each unplaced box.
        # For each unplaced box, we try to find a spot.
        # If no spot, we try to "shake" the system: 
        # Pick a random placed box, try to move it to a new valid spot.
        # If it moves, retry placing the unplaced box.
        
        # Limit time for local search
        ls_start = time.time()
        ls_budget = time_limit - ls_start
        
        for target_idx in unplaced:
            if time.time() > time_limit:
                break
            
            w, h, d = boxes[target_idx]
            orients = box_orientations[target_idx]
            
            # First, check if it fits now (maybe previous moves opened space)
            if not _try_place_box(occupancy, Wi, Hi, Di, target_idx, orients, results, placed_boxes):
                # Try to create space
                # Attempt a few random relocations of existing boxes
                # We pick random placed boxes and try to move them.
                # If a move is valid (doesn't overlap), we keep it.
                # This is a "shake" heuristic.
                
                # To be efficient, we only try this a few times per target box.
                for _ in range(10): # 10 attempts to shake
                    if time.time() > time_limit:
                        break
                    
                    # Pick a random placed box
                    if not placed_boxes:
                        break
                    pos = random.randint(0, len(placed_boxes) - 1)
                    bi, bx, by, bz, bw, bh, bd = placed_boxes[pos]
                    
                    # Try to move this box to a random valid new location
                    # Generate random candidate positions? 
                    # Or scan for a new spot? Scanning is expensive.
                    # Let's try a few random shifts.
                    moved = False
                    for _ in range(5):
                        # Random shift
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)
                        dz = random.randint(-2, 2)
                        nx, ny, nz = bx + dx, by + dy, bz + dz
                        if can_place(nx, ny, nz, bw, bh, bd):
                            # Check if it's the same spot
                            if (nx, ny, nz) == (bx, by, bz):
                                continue
                            # Remove and re-add
                            # We need to update occupancy and placed_boxes
                            occupancy[bx:bx+bw, by:by+bh, bz:bz+bd] = 0
                            if can_place(nx, ny, nz, bw, bh, bd):
                                occupancy[nx:nx+bw, ny:ny+bh, nz:nz+bd] = 1
                                placed_boxes[pos] = (bi, nx, ny, nz, bw, bh, bd)
                                results[bi] = (0, nx, ny, nz, bw, bh, bd)
                                moved = True
                                break
                            else:
                                # Restore
                                occupancy[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                                break
                    
                    # After a successful move, retry placing the target
                    if moved:
                        if _try_place_box(occupancy, Wi, Hi, Di, target_idx, orients, results, placed_boxes):
                            break # Target placed, stop shaking for this target

        return results

    def _try_place_box(occ, Wi, Hi, Di, idx, orients, results, placed_boxes):
        """Attempts to place box idx in any orientation at any valid location (scanning)."""
        for ow, oh, od in orients:
            for z in range(Di - od + 1):
                for y in range(Hi - oh + 1):
                    for x in range(Wi - ow + 1):
                        if np.all(occ[x:x+ow, y:y+oh, z:z+od] == 0):
                            occ[x:x+ow, y:y+oh, z:z+od] = 1
                            placed_boxes.append((idx, x, y, z, ow, oh, od))
                            results[idx] = (0, x, y, z, ow, oh, od)
                            return True
        return False

    def run_list_based_greedy(order, time_limit):
        """Fallback for large dimensions or non-integers."""
        placed = []
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        
        def check_overlap(new_box, placed):
            nx, ny, nz, nw, nh, nd = new_box
            for px, py, pz, pw, ph, pd, _ in placed:
                if not (nx + nw <= px or px + pw <= nx or
                        ny + nh <= py or py + ph <= ny or
                        nz + nd <= pz or pz + pd <= nz):
                    return True
            return False

        for idx in order:
            if time.time() > time_limit:
                break
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            best_pos = None
            best_metric = float('inf')
            
            # Generate candidates: Origin + corners of existing boxes
            candidates = [(0,0,0)]
            for px, py, pz, pw, ph, pd, _ in placed:
                candidates.append((px+pw, py, pz))
                candidates.append((px, py+ph, pz))
                candidates.append((px, py, pz+pd))
            
            for cx, cy, cz in candidates:
                for ow, oh, od in orients:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if check_overlap((cx, cy, cz, ow, oh, od), placed):
                        continue
                    # Metric: minimize max extent
                    metric = max(cx+ow, cy+oh, cz+od)
                    if metric < best_metric:
                        best_metric = metric
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd, idx))
                results[idx] = (0, bx, by, bz, bw, bh, bd)
        
        return results

    # Multi-Start Strategy
    best_results = None
    best_score = 0.0
    
    # Determine number of restarts based on time
    # Grid method is faster per iteration? Scanning is O(WHD). 
    # If W,H,D ~ 50, WHD=125k. 60 boxes * 125k = 7.5M ops. Fast.
    # So we can do many restarts.
    
    max_restarts = 50
    restart_count = 0
    
    while time.time() < deadline - 2.0 and restart_count < max_restarts:
        restart_count += 1
        
        # Create randomized order
        # Strategy: Mostly sorted, but with some shuffling to escape local optima
        current_order = list(indices)
        
        # 20% chance of pure random
        if random.random() < 0.2:
            random.shuffle(current_order)
        else:
            # Perturb sorted order: swap a few random elements
            for _ in range(3):
                i, j = random.sample(range(len(current_order)), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Run packing
        res = run_packing(current_order, deadline)
        
        # Calculate score
        vol = 0
        for r in res:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
        score = vol / container_volume
        
        if score > best_score:
            best_score = score
            best_results = res
            
        # Check time
        if time.time() > deadline - 1.0:
            break

    # If no results found (shouldn't happen), return all -1
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Ensure all boxes are accounted for
    for i in range(len(boxes)):
        if best_results[i] is None:
            w, h, d = boxes[i]
            best_results[i] = (-1, 0, 0, 0, w, h, d)
            
    return best_results
# EVOLVE-BLOCK-END
