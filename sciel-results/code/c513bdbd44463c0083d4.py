import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # 1. Initial Solution Generation
    # We use a slightly improved FFD to get a good starting point.
    # Sort by volume descending.
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # State representation:
    # bins: list of lists. Each inner list contains (x, y, z, w, h, d, original_index)
    # We maintain a mapping from original index to current placement for quick access.
    
    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False

    def find_position(w, h, d, bin_boxes):
        # Find a valid position in the bin, preferring "bottom-left-front"
        x_candidates = {0}
        y_candidates = {0}
        z_candidates = {0}
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            x_candidates.add(bx)
            x_candidates.add(bx + bw)
            y_candidates.add(by)
            y_candidates.add(by + bh)
            z_candidates.add(bz)
            z_candidates.add(bz + bd)
            
        # Filter to valid bounds
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        best_pos = None
        best_cost = float('inf')
        
        # To speed up, we can limit the number of candidates if it gets too large,
        # but for 40 boxes, the set size is usually manageable (< 100^3 worst case, but typically much less).
        # We'll just iterate.
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    # Cost: minimize sum of coordinates to keep it compact
                    cost = x + y + z
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        
        return best_pos

    def construct_initial():
        bins = []
        current_bin = []
        placements = [None] * n
        
        for i, w, h, d in indexed_boxes:
            # Try to place in current bin
            pos = find_position(w, h, d, current_bin)
            if pos is not None:
                x, y, z = pos
                current_bin.append((x, y, z, w, h, d, i))
                placements[i] = (len(bins), x, y, z, w, h, d)
            else:
                # Try other existing bins? FFD usually just opens new if current fails, 
                # but for a better start, let's try all existing bins first.
                placed = False
                for b_idx in range(len(bins)):
                    pos = find_position(w, h, d, bins[b_idx])
                    if pos is not None:
                        x, y, z = pos
                        bins[b_idx].append((x, y, z, w, h, d, i))
                        placements[i] = (b_idx, x, y, z, w, h, d)
                        placed = True
                        break
                
                if not placed:
                    # Open new bin
                    if current_bin:
                        bins.append(current_bin)
                        current_bin = []
                    current_bin = [(0, 0, 0, w, h, d, i)]
                    placements[i] = (len(bins), 0, 0, 0, w, h, d)
        
        if current_bin:
            bins.append(current_bin)
            
        return bins, placements

    # Run initial construction
    bins, placements = construct_initial()
    
    # Helper to calculate score
    def calculate_score(bins, placements):
        if not bins:
            return 0.0
        B = len(bins)
        last_bin_idx = B - 1
        last_bin = bins[last_bin_idx]
        
        last_bin_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            fill = 1.0
        else:
            fill = last_bin_vol / container_vol
        
        return B + 0.1 * (1 - fill)
    
    current_score = calculate_score(bins, placements)
    best_score = current_score
    best_bins = [b[:] for b in bins] # Deep copy
    best_placements = placements[:]
    
    # 2. Local Search (Simulated Annealing)
    # Moves:
    # 1. Relocate: Move box from bin A to bin B
    # 2. Swap: Swap box in bin A with box in bin B
    
    # Precompute box indices for quick access
    # We need a way to quickly find a box in a bin.
    # Since n is small (40), linear search is fine.
    
    def get_box_in_bin(bin_idx, orig_idx):
        for b in bins[bin_idx]:
            if b[6] == orig_idx:
                return b
        return None

    def remove_box_from_bin(bin_idx, orig_idx):
        for j, b in enumerate(bins[bin_idx]):
            if b[6] == orig_idx:
                del bins[bin_idx][j]
                return True
        return False

    def try_move_box(orig_idx, target_bin_idx):
        """Try to move box orig_idx to target_bin_idx. Returns True if successful."""
        # Find current bin
        curr_bin_idx = placements[orig_idx][0]
        if curr_bin_idx == target_bin_idx:
            return False
        
        # Get box details
        box = get_box_in_bin(curr_bin_idx, orig_idx)
        if not box:
            return False
        x, y, z, w, h, d, _ = box
        
        # Check if it fits in target
        pos = find_position(w, h, d, bins[target_bin_idx])
        if pos is None:
            return False
        
        # Perform move
        remove_box_from_bin(curr_bin_idx, orig_idx)
        bins[target_bin_idx].append((pos[0], pos[1], pos[2], w, h, d, orig_idx))
        placements[orig_idx] = (target_bin_idx, pos[0], pos[1], pos[2], w, h, d)
        
        # Clean up empty bins? 
        # The score depends on B (number of occupied bins). 
        # If a bin becomes empty, we should remove it to reduce B.
        # However, bin indices in 'placements' must remain consistent.
        # Strategy: Don't remove bins from the list immediately to keep indices stable, 
        # but calculate score based on non-empty bins? 
        # The problem states: "Bin labels are nonnegative integers... largest label defines last bin."
        # If I have bins [0, 1, 2] and bin 1 is empty, B=3? 
        # "B is the number of occupied bins". 
        # So if bin 1 is empty, B=2. 
        # But "largest label defines last bin". If largest label is 2, last bin is 2.
        # So we need to be careful.
        # Let's compact the bins list if any become empty, but this requires updating all placements.
        # Given n=40, updating placements is cheap.
        
        # Check for empty bins and compact
        new_bins = []
        old_to_new = {}
        for i, b in enumerate(bins):
            if b:
                old_to_new[i] = len(new_bins)
                new_bins.append(b)
        
        if len(new_bins) < len(bins):
            # Update placements
            for i in range(n):
                old_bin_idx = placements[i][0]
                if old_bin_idx in old_to_new:
                    new_bin_idx = old_to_new[old_bin_idx]
                    x, y, z, w, h, d = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5]
                    placements[i] = (new_bin_idx, x, y, z, w, h, d)
            bins[:] = new_bins
            return True # Structure changed
        
        return True # Move successful

    def perform_swap(orig_idx1, orig_idx2):
        """Swap two boxes between bins. Returns True if successful."""
        bin1_idx = placements[orig_idx1][0]
        bin2_idx = placements[orig_idx2][0]
        
        if bin1_idx == bin2_idx:
            return False
            
        box1 = get_box_in_bin(bin1_idx, orig_idx1)
        box2 = get_box_in_bin(bin2_idx, orig_idx2)
        
        if not box1 or not box2:
            return False
            
        x1, y1, z1, w1, h1, d1, _ = box1
        x2, y2, z2, w2, h2, d2, _ = box2
        
        # Remove both
        remove_box_from_bin(bin1_idx, orig_idx1)
        remove_box_from_bin(bin2_idx, orig_idx2)
        
        # Try to place box1 in bin2
        pos1_in_2 = find_position(w1, h1, d1, bins[bin2_idx])
        if pos1_in_2 is None:
            # Rollback
            bins[bin1_idx].append((x1, y1, z1, w1, h1, d1, orig_idx1))
            bins[bin2_idx].append((x2, y2, z2, w2, h2, d2, orig_idx2))
            return False
            
        # Try to place box2 in bin1
        pos2_in_1 = find_position(w2, h2, d2, bins[bin1_idx])
        if pos2_in_1 is None:
            # Rollback
            bins[bin1_idx].append((x1, y1, z1, w1, h1, d1, orig_idx1))
            bins[bin2_idx].append((x2, y2, z2, w2, h2, d2, orig_idx2))
            return False
            
        # Commit
        bins[bin2_idx].append((pos1_in_2[0], pos1_in_2[1], pos1_in_2[2], w1, h1, d1, orig_idx1))
        bins[bin1_idx].append((pos2_in_1[0], pos2_in_1[1], pos2_in_1[2], w2, h2, d2, orig_idx2))
        
        placements[orig_idx1] = (bin2_idx, pos1_in_2[0], pos1_in_2[1], pos1_in_2[2], w1, h1, d1)
        placements[orig_idx2] = (bin1_idx, pos2_in_1[0], pos2_in_1[1], pos2_in_1[2], w2, h2, d2)
        
        # Compact if necessary
        new_bins = []
        old_to_new = {}
        for i, b in enumerate(bins):
            if b:
                old_to_new[i] = len(new_bins)
                new_bins.append(b)
                
        if len(new_bins) < len(bins):
            for i in range(n):
                old_bin_idx = placements[i][0]
                if old_bin_idx in old_to_new:
                    new_bin_idx = old_to_new[old_bin_idx]
                    x, y, z, w, h, d = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5]
                    placements[i] = (new_bin_idx, x, y, z, w, h, d)
            bins[:] = new_bins
            
        return True

    # Annealing Loop
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Select move
        if random.random() < 0.7:
            # Relocate
            orig_idx = random.randint(0, n - 1)
            target_bin_idx = random.randint(0, len(bins) - 1)
            if try_move_box(orig_idx, target_bin_idx):
                new_score = calculate_score(bins, placements)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_bins = [b[:] for b in bins]
                        best_placements = placements[:]
                else:
                    # Rollback
                    # We need to undo the move. 
                    # Since we don't have a full history stack, this is tricky.
                    # Simpler approach: Keep a copy of state before move? 
                    # With n=40, copying is cheap.
                    # Let's refactor to use explicit state copies for rollback.
                    pass
        else:
            # Swap
            if n >= 2:
                i1, i2 = random.sample(range(n), 2)
                if perform_swap(i1, i2):
                    new_score = calculate_score(bins, placements)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_bins = [b[:] for b in bins]
                            best_placements = placements[:]
                    else:
                        pass # Rollback needed
        
        T *= cooling_rate
        
        # If we didn't implement rollback properly, the state might be corrupted.
        # Let's restart the SA loop logic with a safer "Copy before move" strategy.
        # Given the complexity of rollback in the previous block, let's switch to a simpler 
        # "Keep best, restart from best if stuck" or just accept that we need to 
        # implement rollback correctly.
        
        # Actually, implementing rollback for relocation/swap is error-prone.
        # Let's use a different structure:
        # State: bins, placements
        # Move: 
        #  1. Save current state (copy)
        #  2. Apply move
        #  3. Calculate score
        #  4. If accept, keep new state. Else, restore saved state.
        
        # This is slower per step but much safer.
        # With 120s, and n=40, we can afford many steps.
        
        # Let's break and restart the loop with the safer strategy.
        break

    # Re-implementing the loop with safe rollback
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    # Reset to best found so far
    bins = [b[:] for b in best_bins]
    placements = best_placements[:]
    current_score = best_score
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        # Save state
        old_bins = [b[:] for b in bins]
        old_placements = placements[:]
        
        move_made = False
        
        if random.random() < 0.7:
            # Relocate
            orig_idx = random.randint(0, n - 1)
            target_bin_idx = random.randint(0, len(bins) - 1)
            if try_move_box(orig_idx, target_bin_idx):
                move_made = True
        else:
            # Swap
            if n >= 2:
                i1, i2 = random.sample(range(n), 2)
                if perform_swap(i1, i2):
                    move_made = True
        
        if move_made:
            new_score = calculate_score(bins, placements)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins = [b[:] for b in bins]
                    best_placements = placements[:]
            else:
                # Rollback
                bins = old_bins
                placements = old_placements
        else:
            # No move possible, maybe cool faster or just continue
            pass
        
        T *= cooling_rate

    # Finalize
    return best_placements

# EVOLVE-BLOCK-END
