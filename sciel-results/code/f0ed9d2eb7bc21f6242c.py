"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute volumes
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Sort indices by volume descending for initial heuristic
    sorted_indices = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # --- Data Structures ---
    # bins: list of lists of (box_idx, x, y, z)
    # We will maintain this structure for the search
    
    def get_initial_solution():
        bins = []
        
        def find_place_in_bin(bin_items, box_idx):
            w, h, d = boxes[box_idx]
            # Candidate points
            xs = {0}
            ys = {0}
            zs = {0}
            for b_idx, bx, by, bz in bin_items:
                bw, bh, bd = boxes[b_idx]
                if bx + bw <= W: xs.add(bx + bw)
                if by + bh <= H: ys.add(by + bh)
                if bz + bd <= D: zs.add(bz + bd)
            
            valid_xs = sorted([x for x in xs if x + w <= W])
            if not valid_xs: return None
            valid_ys = sorted([y for y in ys if y + h <= H])
            if not valid_ys: return None
            valid_zs = sorted([z for z in zs if z + d <= D])
            if not valid_zs: return None
            
            best_pos = None
            best_score = float('inf')
            
            for x in valid_xs:
                for y in valid_ys:
                    for z in valid_zs:
                        # Check overlap
                        overlap = False
                        for b_idx, bx, by, bz in bin_items:
                            bw, bh, bd = boxes[b_idx]
                            if not (x + w <= bx or bx + bw <= x or
                                    y + h <= by or by + bh <= y or
                                    z + d <= bz or bz + bd <= z):
                                overlap = True
                                break
                        if overlap:
                            continue
                        
                        score = x * (H * D) + y * D + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
            
            return best_pos

        for idx in sorted_indices:
            placed = False
            for b_idx in range(len(bins)):
                pos = find_place_in_bin(bins[b_idx], idx)
                if pos:
                    bins[b_idx].append((idx, pos[0], pos[1], pos[2]))
                    placed = True
                    break
            if not placed:
                # New bin at (0,0,0)
                bins.append([(idx, 0, 0, 0)])
        
        return bins

    def calculate_score(bins):
        if not bins:
            return 0.0
        
        # B is the number of occupied bins
        # "last bin" is the one with the highest index that is occupied?
        # Or just the last bin in the list?
        # The problem says "largest label defines the last bin".
        # In our list representation, if we keep empty bins, the largest label is the last index.
        # But we should compact empty bins to avoid penalty for empty last bin.
        # Let's count occupied bins.
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
        
        # Last bin is the one with the highest index in the original list?
        # If we compact, the last bin is occupied_bins[-1].
        # But the score depends on the "last bin" defined by the largest label.
        # If we output placements, we can choose the labels.
        # So we should map occupied bins to 0, 1, ..., B-1.
        # Then the last bin is the one with label B-1.
        
        # So we need to calculate the volume of the bin that will be assigned the highest label.
        # In our local search, we don't know the final labeling yet.
        # But we can assume we will label them 0..B-1 in order of their index in `bins` (ignoring empties).
        # So the "last bin" is the last non-empty bin in the `bins` list.
        
        last_bin = occupied_bins[-1]
        last_bin_vol = sum(volumes[box_idx] for box_idx, x, y, z in last_bin)
        
        if container_vol == 0:
            return B
        fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # --- Local Search ---
    
    # 1. Get Initial Solution
    current_bins = get_initial_solution()
    current_score = calculate_score(current_bins)
    
    best_bins = [list(b) for b in current_bins]
    best_score = current_score
    
    # Helper to check if a box can be placed at a specific position in a bin
    def can_place_at(bin_items, box_idx, x, y, z, exclude_idx=None):
        w, h, d = boxes[box_idx]
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        for b_idx, bx, by, bz in bin_items:
            if b_idx == exclude_idx:
                continue
            bw, bh, bd = boxes[b_idx]
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    # Helper to find a valid position for a box in a bin (for relocation)
    def find_any_position(bin_items, box_idx):
        w, h, d = boxes[box_idx]
        # Try a few random positions? Or systematic?
        # Systematic is safer for validity.
        xs = {0}
        ys = {0}
        zs = {0}
        for b_idx, bx, by, bz in bin_items:
            bw, bh, bd = boxes[b_idx]
            if bx + bw <= W: xs.add(bx + bw)
            if by + bh <= H: ys.add(by + bh)
            if bz + bd <= D: zs.add(bz + bd)
        
        # Add some random points to break symmetry?
        # For now, just sorted candidates.
        valid_xs = sorted([x for x in xs if x + w <= W])
        if not valid_xs: return None
        valid_ys = sorted([y for y in ys if y + h <= H])
        if not valid_ys: return None
        valid_zs = sorted([z for z in zs if z + d <= D])
        if not valid_zs: return None
        
        for x in valid_xs:
            for y in valid_ys:
                for z in valid_zs:
                    if can_place_at(bin_items, box_idx, x, y, z):
                        return (x, y, z)
        return None

    # Move Types:
    # 1. Relocation: Move box i from bin A to bin B.
    # 2. Swap: Swap box i in bin A with box j in bin B.
    
    # We will use a simple hill-climbing / simulated annealing approach.
    # Given the time limit, we'll do a fixed number of iterations or until time runs out.
    
    # To speed up, we can pre-calculate some things?
    # For now, let's just iterate.
    
    # Randomize the order of boxes in bins for moves?
    # We'll pick random boxes.
    
    max_iterations = 5000
    temperature = 1.0
    cooling = 0.99
    
    for iter in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Relocation
            if len(current_bins) < 2:
                continue
                
            # Pick a random box from a random bin (not the last one, to encourage moving out of last bin? 
            # Or just any bin).
            # Let's pick a box from the last bin first, as that helps reduce B or improve last bin fill.
            # Or pick randomly.
            
            # Pick source bin
            # Bias towards last bin if it's not empty
            if random.random() < 0.5 and current_bins:
                src_bin_idx = len(current_bins) - 1
            else:
                src_bin_idx = random.randint(0, len(current_bins) - 1)
            
            if not current_bins[src_bin_idx]:
                continue
                
            box_pos_in_bin = random.randint(0, len(current_bins[src_bin_idx]) - 1)
            box_idx, bx, by, bz = current_bins[src_bin_idx][box_pos_in_bin]
            
            # Pick destination bin (different from source)
            dest_bin_idx = random.randint(0, len(current_bins) - 1)
            if dest_bin_idx == src_bin_idx:
                continue
            
            # Try to find a position in dest bin
            pos = find_any_position(current_bins[dest_bin_idx], box_idx)
            if pos is None:
                continue
            
            # Apply move
            new_pos = pos
            current_bins[src_bin_idx].pop(box_pos_in_bin)
            if not current_bins[src_bin_idx]:
                # Remove empty bin? 
                # If we remove it, indices shift. This is dangerous.
                # Let's keep empty bins in the list but ignore them in score?
                # No, score depends on B (occupied bins).
                # If we keep empty bins, B is wrong.
                # So we MUST compact or handle indices carefully.
                
                # Strategy: Keep the list fixed size? No, B is dynamic.
                # Better Strategy: Don't remove from list immediately. 
                # Instead, mark as empty? 
                # And in calculate_score, count non-empty bins.
                # And "last bin" is the highest index with non-empty content?
                # The problem says "largest label defines the last bin".
                # If bin 5 is empty, and bin 4 is full, is the last bin 5 (vol 0) or 4?
                # "Bin labels are nonnegative integers... largest label defines the last bin."
                # If I output placements with bin labels 0,1,2,4 (3 is empty), the largest label is 4.
                # So the last bin is 4.
                # So empty bins in the middle don't affect the "last bin" definition if there are higher labels.
                # But if the highest label bin is empty, the last bin is that empty bin?
                # "last-bin occupied volume" -> 0.
                # So yes, we should count occupied bins for B, and the highest occupied bin index for the volume term.
                
                # So, in our list `current_bins`, if a bin becomes empty, we should NOT remove it from the list 
                # if it's not the last one? 
                # Actually, if we remove it, the indices of subsequent bins change.
                # This makes mapping back to original indices hard.
                
                # Alternative: Use a dictionary or fixed-size array?
                # Let's stick to the list but be careful.
                # If a bin becomes empty, we can leave it as an empty list in `current_bins`.
                # `calculate_score` will count `len([b for b in current_bins if b])`.
                # And for last bin, find `max(i for i, b in enumerate(current_bins) if b)`.
                
                pass # Do nothing, leave empty list
            
            current_bins[dest_bin_idx].append((box_idx, new_pos[0], new_pos[1], new_pos[2]))
            
            new_score = calculate_score(current_bins)
            
            # Accept/Reject
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temperature):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins = [list(b) for b in current_bins]
            else:
                # Revert
                current_bins[dest_bin_idx].pop()
                current_bins[src_bin_idx].insert(box_pos_in_bin, (box_idx, bx, by, bz))
                
        else:
            # Swap
            if len(current_bins) < 2:
                continue
                
            # Pick two distinct bins
            b1, b2 = random.sample(range(len(current_bins)), 2)
            if not current_bins[b1] or not current_bins[b2]:
                continue
                
            idx1_pos = random.randint(0, len(current_bins[b1]) - 1)
            idx2_pos = random.randint(0, len(current_bins[b2]) - 1)
            
            box1_idx, x1, y1, z1 = current_bins[b1][idx1_pos]
            box2_idx, x2, y2, z2 = current_bins[b2][idx2_pos]
            
            # Check if box1 fits in bin2 at x2,y2,z2 (excluding box2)
            if not can_place_at(current_bins[b2], box1_idx, x2, y2, z2, exclude_idx=box2_idx):
                continue
            # Check if box2 fits in bin1 at x1,y1,z1 (excluding box1)
            if not can_place_at(current_bins[b1], box2_idx, x1, y1, z1, exclude_idx=box1_idx):
                continue
            
            # Apply Swap
            current_bins[b1][idx1_pos] = (box2_idx, x1, y1, z1)
            current_bins[b2][idx2_pos] = (box1_idx, x2, y2, z2)
            
            new_score = calculate_score(current_bins)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temperature):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins = [list(b) for b in current_bins]
            else:
                # Revert
                current_bins[b1][idx1_pos] = (box1_idx, x1, y1, z1)
                current_bins[b2][idx2_pos] = (box2_idx, x2, y2, z2)
        
        temperature *= cooling
        if temperature < 0.01:
            break

    # Finalize: Compact the bins to ensure valid labels?
    # The problem says labels need not be consecutive.
    # But "largest label defines the last bin".
    # If we have empty bins in the middle, they don't affect B (occupied count).
    # But they DO affect the "largest label" if they are at the end?
    # No, if bin 5 is empty, and bin 4 is full, largest label is 4 (if we consider only occupied bins for the "last bin" definition? 
    # Wait. "Bin labels are nonnegative integers... largest label defines the last bin."
    # This implies we should assign labels such that the largest label corresponds to the bin we want to be the "last bin".
    # If we leave empty bins at the end (high indices), they will be the "last bin" with 0 volume.
    # This is bad for the score (0.1 * (1-0) = 0.1 penalty).
    # So we MUST compact the bins so that the highest index is an occupied bin.
    
    # Re-map indices
    # Create a mapping from old bin index to new bin index
    new_bins = []
    old_to_new = {}
    for i, b in enumerate(best_bins):
        if b:
            old_to_new[i] = len(new_bins)
            new_bins.append(b)
    
    # Generate placements
    placements = [None] * n
    for old_idx, b in enumerate(best_bins):
        if not b:
            continue
        new_idx = old_to_new[old_idx]
        for box_idx, x, y, z in b:
            w, h, d = boxes[box_idx]
            placements[box_idx] = (new_idx, x, y, z, w, h, d)
            
    # Check if any placement is None (should not happen)
    for i in range(n):
        if placements[i] is None:
            # Fallback: put in a new bin
            # This shouldn't happen if the search maintained validity
            w, h, d = boxes[i]
            placements[i] = (len(new_bins), 0, 0, 0, w, h, d)
            new_bins.append([(i, 0, 0, 0)])
            
    return placements

# EVOLVE-BLOCK-END
