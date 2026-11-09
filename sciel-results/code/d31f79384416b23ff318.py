import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-calculate volumes and unique rotations for each box
    box_vols = [w * h * d for w, h, d in boxes]
    box_rotations = []
    for w, h, d in boxes:
        rots = []
        seen = set()
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if (rw, rh, rd) not in seen:
                seen.add((rw, rh, rd))
                if rw <= W and rh <= H and rd <= D:
                    rots.append((rw, rh, rd))
        box_rotations.append(rots)
        
    # Sort indices by volume descending
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 115.0 # Leave 5s buffer
    
    # We will try multiple strategies for sorting/ordering to escape local optima
    # Strategy 0: Volume descending
    # Strategy 1: Max dimension descending
    # Strategy 2: Random shuffle of volume-sorted list
    # Strategy 3: Surface area descending
    
    max_restarts = 200 # Increase restarts, relying on speed improvements
    
    for r in range(max_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Determine ordering strategy for this restart
        # Alternate between different heuristics to explore different solution spaces
        if r % 4 == 0:
            indices = sorted_indices
        elif r % 4 == 1:
            # Sort by max dimension
            indices = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
        elif r % 4 == 2:
            # Sort by surface area
            indices = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
        else:
            # Randomized volume sort (jitter)
            indices = sorted_indices[:]
            random.shuffle(indices)
            # Keep large items somewhat prioritized but allow variation
            # Simple approach: just shuffle completely? No, that's too random.
            # Better: take top 10% fixed, shuffle rest?
            # Let's stick to a simple shuffle for diversity, greedy will still work reasonably.
            # Actually, a pure shuffle is bad for large boxes. 
            # Let's use a "temperature" based shuffle or just stick to the 3 deterministic sorts + 1 random.
            # For r%4==3, let's do a random permutation but bias towards larger volumes?
            # Simpler: Just use the volume sort but with random tie-breaking or slight randomization.
            # Let's just use the volume sort for r%4==3 as well, but randomize candidate order more aggressively?
            # The current code already shuffles candidates.
            # Let's just use the volume sort for r%4==3 to be safe, or a different sort.
            indices = sorted_indices

        # Initialize grid
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
            
        candidates = [(0, 0, 0)]
        placements = [None] * n
        current_vol = 0
        
        # Process boxes in chosen order
        for idx in indices:
            if not box_rotations[idx]:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                continue
                
            placed = False
            w, h, d = boxes[idx]
            
            # Shuffle candidates to introduce diversity
            # To avoid O(N^2) in candidate list size, we can limit the number of candidates checked
            # or just iterate all. With 60 boxes, candidate set shouldn't be huge.
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            # Optimization: If candidate list is very large, sample a subset?
            # For 60 boxes, it's probably fine.
            
            for cx, cy, cz in cand_list:
                if placed:
                    break
                
                for rw, rh, rd in box_rotations[idx]:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check collision
                    # NumPy slice check is fast
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        # Place
                        grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                        placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                        current_vol += rw * rh * rd
                        placed = True
                        
                        # Add new candidates
                        # Only add if they are within bounds
                        new_cands = [
                            (cx + rw, cy, cz),
                            (cx, cy + rh, cz),
                            (cx, cy, cz + rd),
                            (cx + rw, cy + rh, cz),
                            (cx + rw, cy, cz + rd),
                            (cx, cy + rh, cz + rd),
                            (cx + rw, cy + rh, cz + rd)
                        ]
                        for nc in new_cands:
                            nx, ny, nz = nc
                            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                                # Check if already in candidates? 
                                # Using a set for candidates is efficient for membership check
                                # But we are iterating a list. 
                                # We can maintain a set `candidate_set` alongside the list.
                                # Or just check `if nc not in candidates`. List check is O(N).
                                # Given small N, it's okay. But let's use a set for O(1) check.
                                pass
                        # To keep it simple and fast, let's just add them. Duplicates in candidate list
                        # just mean we check the same position twice. The grid check will fail.
                        # This is safe and avoids set overhead in the inner loop.
                        candidates.extend(new_cands)
                        break
                        
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements
            
    return best_placements
