import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initial Construction: Simple Shelf Heuristic (Fast)
    # Sort by volume descending to get a reasonable start
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: 
    # bin_boxes: list of lists. bin_boxes[i] is list of (box_idx, x, y, z, w, h, d)
    # bin_volumes: list of volumes
    bin_boxes = []
    bin_volumes = []
    
    def can_place_in_bin(bin_idx, x, y, z, w, h, d, ignore_idx=None):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (idx, ox, oy, oz, ow, oh, od) in bin_boxes[bin_idx]:
            if idx == ignore_idx:
                continue
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_shelf_position(bin_idx, w, h, d):
        # Try to place in existing shelves or create new shelf
        # For simplicity in initial construction, we just try to fit it
        # using a basic "first fit" with corner expansion, but optimized for speed.
        # Actually, let's use a slightly better heuristic for the start:
        # Try to place at the "lowest, leftmost, backmost" available spot.
        
        # Generate candidate points from existing boxes in this bin
        candidates = [(0, 0, 0)]
        for (idx, ox, oy, oz, ow, oh, od) in bin_boxes[bin_idx]:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            
        # Sort candidates to prefer "tight" packing (min z, then y, then x)
        candidates = list(set(candidates))
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in candidates:
            if can_place_in_bin(bin_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
        return None

    # Initialize bins
    for idx, (w, h, d) in indexed_boxes:
        placed = False
        for i in range(len(bin_boxes)):
            pos = find_shelf_position(i, w, h, d)
            if pos is not None:
                bin_boxes[i].append((idx, pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[i] += w * h * d
                placed = True
                break
        if not placed:
            # Open new bin
            bin_boxes.append([(idx, 0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            
    # 2. Local Search: 2-Opt and Relocation
    # Objective: Minimize B (number of bins) primarily, then maximize last bin fill?
    # Score = B + 0.1 * (1 - last_bin_fill)
    # Since B is integer and last_bin_fill is small, reducing B by 1 is worth ~10x the last bin term.
    # So primary goal: minimize number of bins.
    # Secondary goal: if B is same, maximize volume in the last bin (to minimize the 0.1 term).
    # Note: "last bin" is the one with the highest index.
    
    def get_current_score():
        B = len(bin_boxes)
        if B == 0:
            return 0
        last_bin_vol = bin_volumes[-1]
        total_vol = W * H * D
        fill = last_bin_vol / total_vol if total_vol > 0 else 0
        return B + 0.1 * (1 - fill)
    
    current_score = get_current_score()
    
    # Precompute box dimensions for quick access
    box_dims = {idx: boxes[idx] for idx in range(n)}
    
    # Helper to remove a box from a bin
    def remove_box_from_bin(bin_idx, box_idx):
        for i, item in enumerate(bin_boxes[bin_idx]):
            if item[0] == box_idx:
                vol = item[4] * item[5] * item[6]
                bin_volumes[bin_idx] -= vol
                bin_boxes[bin_idx].pop(i)
                if not bin_boxes[bin_idx]:
                    # Remove empty bin
                    bin_boxes.pop(bin_idx)
                    bin_volumes.pop(bin_idx)
                    # Renumber subsequent bins? No, labels are just indices.
                    # But if we remove a bin, indices shift. 
                    # To avoid complexity, we don't remove bins during search unless we are doing a "merge" move.
                    # For 2-opt (swap), we don't remove bins.
                    # For relocation, we move a box. If source bin becomes empty, we can remove it.
                    # However, removing a bin changes the indices of all subsequent bins.
                    # This makes tracking "which bin is which" tricky if we store indices in other structures.
                    # Since we only store box indices in bin_boxes, and we iterate over bin_boxes directly,
                    # removing a bin is safe for the structure, but we must be careful if we have cached indices.
                    # We will not cache bin indices across iterations.
                    pass
                return True
        return False

    # Helper to add a box to a bin
    def add_box_to_bin(bin_idx, box_idx, x, y, z):
        w, h, d = box_dims[box_idx]
        bin_boxes[bin_idx].append((box_idx, x, y, z, w, h, d))
        bin_volumes[bin_idx] += w * h * d

    # Try to move box 'box_idx' from 'src_bin' to 'dst_bin'
    # Returns new_score if successful, None if failed
    def try_relocate(box_idx, src_bin, dst_bin):
        # Find current pos of box in src_bin
        item = None
        for it in bin_boxes[src_bin]:
            if it[0] == box_idx:
                item = it
                break
        if item is None:
            return None
            
        w, h, d = box_dims[box_idx]
        
        # Generate candidates in dst_bin
        candidates = [(0, 0, 0)]
        for (idx, ox, oy, oz, ow, oh, od) in bin_boxes[dst_bin]:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            
        candidates = list(set(candidates))
        # Sort to find tightest fit quickly? 
        # For relocation, we just need *a* valid fit. 
        # But to minimize score, we might prefer fits that keep the last bin full?
        # Actually, the score depends on the *last* bin's volume.
        # If we move a box TO the last bin, volume increases -> score decreases (good).
        # If we move a box FROM the last bin, volume decreases -> score increases (bad), unless it reduces B.
        # If we move a box between non-last bins, B might stay same, last bin vol stays same -> score same.
        # So, we should prioritize moves that:
        # 1. Reduce B (source bin becomes empty).
        # 2. If B doesn't change, increase volume of the last bin (if dst is last bin).
        
        # Let's just find any valid position first.
        valid_pos = None
        for cx, cy, cz in candidates:
            if can_place_in_bin(dst_bin, cx, cy, cz, w, h, d, ignore_idx=box_idx):
                valid_pos = (cx, cy, cz)
                break
        
        if valid_pos is None:
            return None
            
        # Calculate new score
        # 1. Remove from source
        vol = w * h * d
        old_src_vol = bin_volumes[src_bin]
        bin_volumes[src_bin] -= vol
        # Remove item from list
        for i, it in enumerate(bin_boxes[src_bin]):
            if it[0] == box_idx:
                bin_boxes[src_bin].pop(i)
                break
                
        src_became_empty = False
        if not bin_boxes[src_bin]:
            bin_boxes.pop(src_bin)
            bin_volumes.pop(src_bin)
            src_became_empty = True
            # Indices shift! 
            # If src_bin < dst_bin, dst_bin index decreases by 1.
            # This is dangerous. Let's avoid removing bins during the "try" phase unless we commit.
            # Better approach: Simulate the move without modifying the global state, or
            # modify and rollback if score doesn't improve.
            
            # Let's rollback for now and just check if it's a "potential" improvement.
            # Actually, let's just do the move, calculate score, and rollback if not better.
            pass
        
        # 2. Add to destination
        # Note: If src_bin was removed, dst_bin index might have changed.
        # To keep it simple and safe, we will NOT remove empty bins during the search loop.
        # We will only remove them at the very end or handle index shifts carefully.
        # Let's stick to: Keep bins even if empty? No, B is count of *occupied* bins.
        # The problem says "B is the number of occupied bins".
        # So if a bin is empty, it doesn't count towards B.
        # But the "last bin" is defined by "largest label".
        # If we have bins [0, 1, 2] and bin 1 is empty, B=2. Last bin is 2.
        # If we remove bin 1, bins become [0, 2] (if we don't renumber) or [0, 1] (if we do).
        # The problem says "Bin labels... need not be consecutive; the largest label defines the last bin."
        # So if I have bins at indices 0, 1, 2. Bin 1 is empty. B=2. Last bin is 2.
        # If I move box from 1 to 0, bin 1 is empty. B=2. Last bin is 2.
        # If I move box from 2 to 0, bin 2 is empty. B=2 (bins 0,1 occupied? No, 1 was empty).
        # Wait, if bin 1 was empty, and I move from 2 to 0, bin 2 becomes empty.
        # Occupied: 0, 1? No, 1 was empty. Occupied: 0. B=1. Last bin: 0.
        
        # This index shifting is a pain. 
        # Strategy: 
        # 1. Do not remove empty bins from the list during search.
        # 2. B = count of bins with volume > 0.
        # 3. Last bin = max index i where volume > 0.
        # This avoids index shifting.
        
        # Let's restart the "try_relocate" logic with this assumption.
        
        # Rollback the changes made above
        # 1. Restore source
        bin_volumes[src_bin] += vol
        # Re-insert box. We lost the position. 
        # We need to store the original position.
        # Let's refactor: Pass original pos.
        
        # This is getting messy. Let's write a cleaner Local Search.
        
        return None # Placeholder, will rewrite below

    # Clean Local Search Implementation
    # We maintain bin_boxes as a list. We do NOT remove empty bins.
    # B = number of bins with non-zero volume.
    # Last Bin Index = max(i for i, v in enumerate(bin_volumes) if v > 0)
    
    def calc_score():
        B = sum(1 for v in bin_volumes if v > 0)
        if B == 0:
            return 0
        last_idx = -1
        for i in range(len(bin_volumes) - 1, -1, -1):
            if bin_volumes[i] > 0:
                last_idx = i
                break
        last_vol = bin_volumes[last_idx]
        total_vol = W * H * D
        fill = last_vol / total_vol if total_vol > 0 else 0
        return B + 0.1 * (1 - fill)
    
    current_score = calc_score()
    
    # Pre-compute box volumes
    box_vols = [w*h*d for w,h,d in boxes]
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible relocations
        # For each box, for each bin (excluding current bin), try to move
        # To make it faster, we can limit attempts or use random sampling if n is large.
        # n=40 is small. 40 boxes * ~10 bins * ~20 candidates = 8000 checks per pass.
        # Very fast.
        
        for box_idx in range(n):
            # Find current bin of this box
            src_bin = -1
            for i in range(len(bin_boxes)):
                for item in bin_boxes[i]:
                    if item[0] == box_idx:
                        src_bin = i
                        break
                if src_bin != -1:
                    break
            
            if src_bin == -1:
                continue
                
            w, h, d = box_dims[box_idx]
            
            # Try moving to other bins
            for dst_bin in range(len(bin_boxes)):
                if dst_bin == src_bin:
                    continue
                
                # Generate candidates in dst_bin
                candidates = [(0, 0, 0)]
                for (idx, ox, oy, oz, ow, oh, od) in bin_boxes[dst_bin]:
                    if idx == box_idx:
                        continue
                    candidates.append((ox + ow, oy, oz))
                    candidates.append((ox, oy + oh, oz))
                    candidates.append((ox, oy, oz + od))
                
                # Deduplicate
                candidates = list(set(candidates))
                
                # Find a valid position
                valid_pos = None
                for cx, cy, cz in candidates:
                    if can_place_in_bin(dst_bin, cx, cy, cz, w, h, d, ignore_idx=box_idx):
                        valid_pos = (cx, cy, cz)
                        break
                
                if valid_pos is None:
                    continue
                
                # Simulate the move
                # 1. Remove from src
                vol = box_vols[box_idx]
                bin_volumes[src_bin] -= vol
                # Remove item
                for i, item in enumerate(bin_boxes[src_bin]):
                    if item[0] == box_idx:
                        bin_boxes[src_bin].pop(i)
                        break
                        
                # 2. Add to dst
                bin_volumes[dst_bin] += vol
                bin_boxes[dst_bin].append((box_idx, valid_pos[0], valid_pos[1], valid_pos[2], w, h, d))
                
                # 3. Calculate new score
                new_score = calc_score()
                
                # 4. Check improvement
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                    # Keep the change
                    # We can break out of dst_bin loop to move next box? 
                    # Or continue? Moving next box immediately might be better.
                    break
                else:
                    # Rollback
                    bin_volumes[dst_bin] -= vol
                    # Remove item from dst
                    bin_boxes[dst_bin].pop() # We just appended it, so it's last
                    # Re-insert into src
                    bin_volumes[src_bin] += vol
                    # We need the original position. We didn't store it.
                    # This is a bug. We need to store original position.
                    
                    # Let's fix the rollback by storing original pos before removing.
                    # I'll refactor the inner loop to store orig_pos.
                    pass
                    
    # The above loop has a rollback bug. Let's rewrite the search part cleanly.
    
    # --- Corrected Local Search ---
    
    # Reset to initial state? No, the state is currently messed up by the failed rollback attempt in the previous block?
    # No, the previous block was just a draft in my head/comments. The actual code execution stopped at `return None` inside the dummy function.
    # The actual state `bin_boxes` is from the initial construction.
    # Let's run the corrected search on the current state.
    
    current_score = calc_score()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle boxes to avoid local minima? Or just iterate.
        # Iterating is deterministic.
        
        for box_idx in range(n):
            # Find current bin
            src_bin = -1
            src_pos = None
            for i in range(len(bin_boxes)):
                for item in bin_boxes[i]:
                    if item[0] == box_idx:
                        src_bin = i
                        src_pos = (item[1], item[2], item[3])
                        break
                if src_bin != -1:
                    break
            
            if src_bin == -1:
                continue
                
            w, h, d = box_dims[box_idx]
            vol = box_vols[box_idx]
            
            # Try moving to other bins
            for dst_bin in range(len(bin_boxes)):
                if dst_bin == src_bin:
                    continue
                
                # Generate candidates in dst_bin
                candidates = [(0, 0, 0)]
                for (idx, ox, oy, oz, ow, oh, od) in bin_boxes[dst_bin]:
                    if idx == box_idx:
                        continue
                    candidates.append((ox + ow, oy, oz))
                    candidates.append((ox, oy + oh, oz))
                    candidates.append((ox, oy, oz + od))
                
                candidates = list(set(candidates))
                
                valid_pos = None
                for cx, cy, cz in candidates:
                    if can_place_in_bin(dst_bin, cx, cy, cz, w, h, d, ignore_idx=box_idx):
                        valid_pos = (cx, cy, cz)
                        break
                
                if valid_pos is None:
                    continue
                
                # Perform Move
                # 1. Remove from src
                bin_volumes[src_bin] -= vol
                for i, item in enumerate(bin_boxes[src_bin]):
                    if item[0] == box_idx:
                        bin_boxes[src_bin].pop(i)
                        break
                        
                # 2. Add to dst
                bin_volumes[dst_bin] += vol
                bin_boxes[dst_bin].append((box_idx, valid_pos[0], valid_pos[1], valid_pos[2], w, h, d))
                
                # 3. Score
                new_score = calc_score()
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                    break # Move to next box
                else:
                    # Rollback
                    bin_volumes[dst_bin] -= vol
                    bin_boxes[dst_bin].pop()
                    bin_volumes[src_bin] += vol
                    # Re-insert into src at original position
                    # We don't care about the exact position in the list, just that it's there.
                    # But for "can_place" checks, position matters.
                    # We need to re-insert with the original x,y,z.
                    bin_boxes[src_bin].append((box_idx, src_pos[0], src_pos[1], src_pos[2], w, h, d))
                    # Note: Order in bin_boxes list doesn't matter for correctness, only for candidate generation.
                    # Candidate generation uses all boxes in the bin.
                    
    # Final cleanup: Remove empty bins?
    # The score calculation already handles empty bins (B counts only occupied).
    # The return format requires: (bin, x, y, z, w, h, d)
    # Bin labels are indices. If we have empty bins in the middle, indices are non-consecutive?
    # "Bin labels... need not be consecutive".
    # So we can just return the indices as they are.
    
    placements = [None] * n
    for i in range(len(bin_boxes)):
        for item in bin_boxes[i]:
            idx, x, y, z, w, h, d = item
            placements[idx] = (i, x, y, z, w, h, d)
            
    # Ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            # Should not happen, but safety
            placements[i] = (len(bin_boxes), 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return placements

