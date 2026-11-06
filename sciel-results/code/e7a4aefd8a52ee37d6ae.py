import time
import random
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-process boxes: (index, w, h, d, volume)
    box_data = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    container_vol = W * H * D
    
    # Sort indices by different heuristics for multi-start
    heuristics = [
        lambda i: box_data[i][4],  # Volume
        lambda i: max(box_data[i][1], box_data[i][2], box_data[i][3]), # Max Dim
        lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3] / max(1, box_data[i][1]+box_data[i][2]+box_data[i][3]), # Volume/Perimeter-ish
        lambda i: -box_data[i][1] - box_data[i][2] - box_data[i][3] # Smallest first (for filling gaps)
    ]
    
    # State representation for a single packing attempt
    # We store placements as a list of tuples: (bin_id, x, y, z, w, h, d)
    # And a structure to quickly check overlaps.
    
    def get_initial_packing(order):
        """
        Greedily pack boxes in the given order.
        Returns:
        - placements: list of (bin, x, y, z, w, h, d) for each box index
        - bins_data: list of lists of placed boxes per bin (for overlap check)
        - num_bins: int
        - last_bin_vol: float
        """
        placements = [None] * n
        bins = [] # List of lists of (x, y, z, w, h, d)
        
        def new_bin():
            return []
        
        bins.append(new_bin())
        
        # For speed, we will use a simple list of corners per bin
        # But for local search, we need to check against ALL placed boxes in a bin.
        # With n=40, checking 40 boxes is fast.
        
        for idx in order:
            w, h, d, vol = box_data[idx][1], box_data[idx][2], box_data[idx][3], box_data[idx][4]
            
            best_bin_idx = -1
            best_pos = None
            best_score = float('inf')
            
            # Try existing bins
            for b_idx in range(len(bins)):
                bin_boxes = bins[b_idx]
                
                # Generate candidate points: corners of existing boxes + (0,0,0)
                # Optimization: If bin is empty, only (0,0,0)
                if not bin_boxes:
                    candidates = [(0,0,0)]
                else:
                    candidates = set()
                    candidates.add((0,0,0))
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        # 7 corners of each box
                        candidates.add((bx+bw, by, bz))
                        candidates.add((bx, by+bh, bz))
                        candidates.add((bx, by, bz+bd))
                        candidates.add((bx+bw, by+bh, bz))
                        candidates.add((bx+bw, by, bz+bd))
                        candidates.add((bx, by+bh, bz+bd))
                        candidates.add((bx+bw, by+bh, bz+bd))
                    
                    # Limit candidates to avoid explosion (though for n=40 it's small)
                    # We can also add points aligned with box edges? 
                    # For now, just corners.
                    
                for cx, cy, cz in candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    # Check overlap
                    overlap = False
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if cx + w <= bx or bx + bw <= cx:
                            continue
                        if cy + h <= by or by + bh <= cy:
                            continue
                        if cz + d <= bz or bz + bd <= cz:
                            continue
                        overlap = True
                        break
                    
                    if overlap:
                        continue
                    
                    # Score: compactness
                    score = cx + cy + cz
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                # New bin
                bins.append(new_bin())
                best_bin_idx = len(bins) - 1
                best_pos = (0, 0, 0)
            
            px, py, pz = best_pos
            placements[idx] = (best_bin_idx, px, py, pz, w, h, d)
            bins[best_bin_idx].append((px, py, pz, w, h, d))
            
        # Calculate score components
        num_bins = len(bins)
        if num_bins > 0:
            last_bin_vol = sum(w*h*d for x,y,z,w,h,d in bins[-1])
        else:
            last_bin_vol = 0
            
        return placements, bins, num_bins, last_bin_vol

    def calc_score(num_bins, last_bin_vol):
        return num_bins + 0.1 * (1 - last_bin_vol / container_vol)

    best_placements = None
    best_score = float('inf')
    
    # 1. Multi-Start
    for h_func in heuristics:
        if time.time() > deadline - 5.0:
            break
            
        order = sorted(range(n), key=h_func, reverse=True)
        # Add some randomness to break ties?
        # Actually, let's just run the deterministic sort.
        
        placements, bins, num_bins, last_vol = get_initial_packing(order)
        score = calc_score(num_bins, last_vol)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            # We need to keep bins structure for local search
            # But best_placements is just the output. 
            # For local search, we need to reconstruct bins from placements or keep them.
            # Let's keep the bins structure associated with the best solution.
            best_bins = bins
            
    # 2. Local Search on the best solution found
    # We will try to improve the packing by moving boxes.
    # Moves:
    # 1. Swap positions of two boxes in the same bin.
    # 2. Move a box to a different position in the same bin (try nearby corners).
    # 3. Move a box to a different bin (if it fits).
    
    # To make this efficient, we'll define a helper to check if a box fits at a pos in a bin
    def fits_in_bin(box_idx, bin_idx, x, y, z, bins):
        w, h, d = box_data[box_idx][1], box_data[box_idx][2], box_data[box_idx][3]
        if x + w > W or y + h > H or z + d > D:
            return False
        
        bin_boxes = bins[bin_idx]
        # Exclude the box itself if it's already there
        for i, (bx, by, bz, bw, bh, bd) in enumerate(bin_boxes):
            # We need to know if this is the box we are moving.
            # This is tricky if we just have coordinates.
            # Better approach: 
            # When checking a move, we temporarily remove the box from its old bin,
            # check the new bin, and if it fits, update.
            pass
        
        # For simplicity in the loop below, we will pass the list of *other* boxes.
        return True # Placeholder, logic handled in loop

    # Let's implement a specific local search step:
    # Pick a random box.
    # Try to move it to a better position in its current bin (check all corners of other boxes).
    # Or try to move it to another bin (check all corners).
    
    # To speed up, we will pre-compute corner candidates for each bin dynamically.
    
    current_placements = best_placements[:]
    current_bins = [b[:] for b in best_bins] # Deep copy lists of tuples
    current_score = best_score
    current_num_bins = len(current_bins)
    current_last_vol = sum(w*h*d for x,y,z,w,h,d in current_bins[-1]) if current_bins else 0
    
    T = 0.5
    cooling = 0.99
    min_T = 0.01
    
    # Helper to update bins structure when moving a box
    def remove_box_from_bin(bin_idx, box_idx, bins, placements):
        # Find the box in the bin
        # We need to identify which entry in bins[bin_idx] corresponds to box_idx
        # This is inefficient if we don't track indices.
        # Let's change representation:
        # bins[b] = list of (box_idx, x, y, z, w, h, d)
        
        # Re-implementing state representation for local search to be index-aware
        pass

    # --- Revised Local Search Implementation ---
    # State:
    # bin_contents: list of lists of (box_idx, x, y, z, w, h, d)
    # placements: list of (bin_idx, x, y, z, w, h, d) for each box_idx
    
    # Convert best_bins (which was list of (x,y,z,w,h,d)) to index-aware
    # We need to map back.
    # Since get_initial_packing didn't store indices in bins, we lost that info.
    # Let's re-run the best initial packing with index-aware bins.
    
    # Find the order that produced best_placements? 
    # We didn't store it. 
    # Let's just re-run the best heuristic with index-aware structure.
    
    # Identify best heuristic
    best_h_func = None
    # Since we iterated, we can just re-run the first one that gave best score?
    # No, we should store the order.
    # Let's just re-run the Volume sort (usually best) with index-aware bins.
    
    order = sorted(range(n), key=lambda i: box_data[i][4], reverse=True)
    
    def get_initial_packing_indexed(order):
        placements = [None] * n
        bins = [] # List of lists of (box_idx, x, y, z, w, h, d)
        
        def new_bin():
            return []
        
        bins.append(new_bin())
        
        for idx in order:
            w, h, d, vol = box_data[idx][1], box_data[idx][2], box_data[idx][3], box_data[idx][4]
            
            best_bin_idx = -1
            best_pos = None
            best_score = float('inf')
            
            for b_idx in range(len(bins)):
                bin_boxes = bins[b_idx]
                
                if not bin_boxes:
                    candidates = [(0,0,0)]
                else:
                    candidates = set([(0,0,0)])
                    for b_idx2, bx, by, bz, bw, bh, bd in bin_boxes:
                        candidates.add((bx+bw, by, bz))
                        candidates.add((bx, by+bh, bz))
                        candidates.add((bx, by, bz+bd))
                        candidates.add((bx+bw, by+bh, bz))
                        candidates.add((bx+bw, by, bz+bd))
                        candidates.add((bx, by+bh, bz+bd))
                        candidates.add((bx+bw, by+bh, bz+bd))
                    
                for cx, cy, cz in candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    overlap = False
                    for b_idx2, bx, by, bz, bw, bh, bd in bin_boxes:
                        if cx + w <= bx or bx + bw <= cx:
                            continue
                        if cy + h <= by or by + bh <= cy:
                            continue
                        if cz + d <= bz or bz + bd <= cz:
                            continue
                        overlap = True
                        break
                    
                    if overlap:
                        continue
                    
                    score = cx + cy + cz
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                bins.append(new_bin())
                best_bin_idx = len(bins) - 1
                best_pos = (0, 0, 0)
            
            px, py, pz = best_pos
            placements[idx] = (best_bin_idx, px, py, pz, w, h, d)
            bins[best_bin_idx].append((idx, px, py, pz, w, h, d))
            
        num_bins = len(bins)
        last_bin_vol = sum(w*h*d for _, x,y,z,w,h,d in bins[-1]) if bins else 0
        return placements, bins, num_bins, last_bin_vol

    # Get the best initial solution again with indexed bins
    # We assume the volume sort is the best one we found in the previous loop.
    # If not, we might have a mismatch, but for now, we'll use the volume sort result 
    # and compare it to the un-indexed best score. If it's worse, we stick to the un-indexed one 
    # and skip local search (or run a simpler one).
    # Actually, let's just run local search on the Volume Sort result, as it's the most robust.
    
    placements, bins, num_bins, last_vol = get_initial_packing_indexed(order)
    score = calc_score(num_bins, last_vol)
    
    # If this is worse than our previous best, we should probably use the previous best.
    # But the previous best didn't have indexed bins.
    # Let's assume the Volume Sort is good enough to start local search.
    # To be safe, we'll run local search on this, and if we don't improve, we return the 
    # best of (previous best, local search result).
    
    best_local_placements = placements
    best_local_score = score
    best_local_bins = bins
    
    # Local Search Loop
    # We will try to improve by moving boxes.
    # Move types:
    # 1. Reposition a box in its current bin to a different valid corner.
    # 2. Move a box to a different bin (if it fits better or reduces bin count? 
    #    Moving to another bin doesn't reduce bin count unless we close the old one, 
    #    which is hard to check dynamically).
    #    Let's focus on Repositioning first, as it improves last_bin_fill and potentially 
    #    allows other boxes to fit later (though order is fixed).
    #    Actually, repositioning doesn't change bin count, only last_bin_vol.
    #    To reduce bin count, we need to "compact" boxes to allow a box from a later bin 
    #    to move into an earlier bin. This is complex.
    #    
    #    Alternative: Simulated Annealing on the PLACEMENTS.
    #    Move: Pick a box, try to move it to a random valid position in a random bin.
    #    Accept if score improves or with probability exp(-delta/T).
    
    current_placements = best_local_placements[:]
    current_bins = [b[:] for b in best_local_bins]
    current_score = best_local_score
    current_num_bins = len(current_bins)
    current_last_vol = sum(w*h*d for _, x,y,z,w,h,d in current_bins[-1]) if current_bins else 0
    
    T = 0.1
    cooling = 0.999
    min_T = 0.001
    
    def get_valid_positions(box_idx, bin_idx, bins, exclude_self=True):
        w, h, d = box_data[box_idx][1], box_data[box_idx][2], box_data[box_idx][3]
        bin_boxes = bins[bin_idx]
        
        candidates = set([(0,0,0)])
        for b_idx2, bx, by, bz, bw, bh, bd in bin_boxes:
            if exclude_self and b_idx2 == box_idx:
                continue
            candidates.add((bx+bw, by, bz))
            candidates.add((bx, by+bh, bz))
            candidates.add((bx, by, bz+bd))
            candidates.add((bx+bw, by+bh, bz))
            candidates.add((bx+bw, by, bz+bd))
            candidates.add((bx, by+bh, bz+bd))
            candidates.add((bx+bw, by+bh, bz+bd))
            
        valid = []
        for cx, cy, cz in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            overlap = False
            for b_idx2, bx, by, bz, bw, bh, bd in bin_boxes:
                if exclude_self and b_idx2 == box_idx:
                    continue
                if cx + w <= bx or bx + bw <= cx:
                    continue
                if cy + h <= by or by + bh <= cy:
                    continue
                if cz + d <= bz or bz + bd <= cz:
                    continue
                overlap = True
                break
            if not overlap:
                valid.append((cx, cy, cz))
        return valid

    while time.time() < deadline - 5.0 and T > min_T:
        if random.random() < 0.5:
            # Reposition in same bin
            idx = random.randint(0, n-1)
            cur_bin = current_placements[idx][0]
            
            # Get current pos
            _, cx, cy, cz, w, h, d = current_placements[idx]
            
            # Get valid positions in this bin
            valid_pos = get_valid_positions(idx, cur_bin, current_bins, exclude_self=True)
            
            if not valid_pos:
                T *= cooling
                continue
                
            # Pick a random valid position different from current
            if (cx, cy, cz) in valid_pos:
                valid_pos.remove((cx, cy, cz))
            
            if not valid_pos:
                T *= cooling
                continue
                
            new_pos = random.choice(valid_pos)
            
            # Check if this move changes the "last bin volume" or "num bins".
            # Repositioning within a bin does NOT change num_bins.
            # It DOES change last_bin_vol ONLY if the bin is the last bin.
            # Wait, "last bin" is defined by the largest label.
            # If we move a box within the last bin, the volume of that bin stays the same.
            # If we move a box OUT of the last bin into an earlier bin, the last bin volume DECREASES.
            # If we move a box INTO the last bin from an earlier bin, the last bin volume INCREASES.
            # 
            # Score = B + 0.1 * (1 - LastBinVol / ContainerVol)
            # Lower is better.
            # So we want to DECREASE LastBinVol.
            # This implies we want to move boxes OUT of the last bin if possible, 
            # or keep the last bin full? 
            # Wait. (1 - LastBinVol / ContainerVol).
            # If LastBinVol is HIGH, (1 - High) is LOW. Score is LOW. Good.
            # If LastBinVol is LOW, (1 - Low) is HIGH. Score is HIGH. Bad.
            # So we want the LAST BIN TO BE AS FULL AS POSSIBLE.
            
            # So, if we move a box from an earlier bin to the last bin, LastBinVol increases -> Score decreases (improves).
            # If we move a box from the last bin to an earlier bin, LastBinVol decreases -> Score increases (worsens).
            # UNLESS moving it to an earlier bin allows us to CLOSE the last bin?
            # No, closing the bin requires moving ALL boxes out.
            
            # So, simple repositioning within the same bin doesn't change volume.
            # It only helps if it allows OTHER boxes to be packed better? 
            # But we are not re-packing. We are just moving one box.
            # So repositioning a single box in a fixed set of bins doesn't change the score 
            # unless it changes which bin is the "last bin" (by changing labels? No, labels are fixed).
            # Or if it changes the volume of the last bin (by moving in/out).
            
            # Therefore, simple repositioning is USELESS for the score unless it enables a 
            # subsequent move that reduces bin count.
            # 
            # So we need moves that change bin count or last bin volume.
            # 
            # Move: Try to move a box from Bin X to Bin Y.
            # If Y < X, and Bin X becomes empty, B decreases. Score improves by 1.
            # If Y > X, B might increase (if Y is new) or stay same. LastBinVol changes.
            
            # Let's try: Pick a box from the last bin. Try to move it to an earlier bin.
            # If it fits, and the last bin becomes empty, B decreases.
            # If it fits, but last bin is not empty, LastBinVol decreases (bad).
            
            # So, we should ONLY move boxes from the last bin to earlier bins if it empties the last bin.
            # Or, we should move boxes from earlier bins to the last bin to fill it up?
            # Filling the last bin up improves the 0.1 term.
            
            # Strategy:
            # 1. Try to fill the last bin by moving boxes from earlier bins into it.
            # 2. Try to empty the last bin by moving boxes from it to earlier bins (only if it empties).
            
            # Let's try moving a box from an earlier bin to the last bin.
            if current_num_bins > 1:
                last_bin_idx = current_num_bins - 1
                # Pick a box from an earlier bin
                src_bin = random.randint(0, last_bin_idx - 1)
                if not current_bins[src_bin]:
                    T *= cooling
                    continue
                box_in_src = random.choice(current_bins[src_bin])
                idx = box_in_src[0]
                
                # Try to place this box in the last bin
                valid_pos = get_valid_positions(idx, last_bin_idx, current_bins, exclude_self=False)
                if valid_pos:
                    new_pos = random.choice(valid_pos)
                    
                    # Execute move
                    # Remove from src_bin
                    current_bins[src_bin].remove(box_in_src)
                    # Add to last_bin
                    current_bins[last_bin_idx].append((idx, new_pos[0], new_pos[1], new_pos[2], w, h, d))
                    # Update placement
                    current_placements[idx] = (last_bin_idx, new_pos[0], new_pos[1], new_pos[2], w, h, d)
                    
                    # Check if src_bin is now empty
                    if not current_bins[src_bin]:
                        # We could remove the bin, but labels are non-consecutive?
                        # "Bin labels are nonnegative integers and need not be consecutive; 
                        # the largest label defines the last bin."
                        # If we remove a bin, do we relabel?
                        # The score depends on B (number of occupied bins).
                        # If we remove an empty bin, B decreases.
                        # We can just leave it as an empty bin in the list, and count non-empty ones.
                        pass
                    
                    # Calculate new score
                    new_num_bins = sum(1 for b in current_bins if b)
                    new_last_bin_idx = max(i for i, b in enumerate(current_bins) if b)
                    new_last_vol = sum(w*h*d for _, x,y,z,w,h,d in current_bins[new_last_bin_idx])
                    new_score = calc_score(new_num_bins, new_last_vol)
                    
                    delta = new_score - current_score
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_score = new_score
                        current_num_bins = new_num_bins
                        current_last_vol = new_last_vol
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local_placements = current_placements[:]
                            best_local_bins = [b[:] for b in current_bins]
                    else:
                        # Revert
                        current_bins[last_bin_idx].remove((idx, new_pos[0], new_pos[1], new_pos[2], w, h, d))
                        current_bins[src_bin].append(box_in_src)
                        current_placements[idx] = (src_bin, box_in_src[1], box_in_src[2], box_in_src[3], w, h, d)
            
            else:
                # Only one bin, can't move between bins.
                # Try to compact? But score doesn't change with compaction if B=1 and LastBinVol is fixed.
                # Wait, if B=1, LastBinVol is the total volume. It's fixed.
                # So if B=1, score is 1 + 0.1*(1 - TotalVol/ContainerVol).
                # This is constant! No search needed.
                break

        T *= cooling

    # Final Result
    # Compare best_local_placements with the initial un-indexed best_placements
    # We need to ensure best_local_placements is valid.
    
    # Convert best_local_placements to the required format
    # It is already a list of (bin, x, y, z, w, h, d)
    
    # Safety check: ensure all boxes are placed
    for i in range(n):
        if best_local_placements[i] is None:
            # Fallback
            best_local_placements = best_placements # Use the initial one
            break
    
    return best_local_placements

