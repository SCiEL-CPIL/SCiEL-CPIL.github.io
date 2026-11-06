import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    N = 256

    # Precompute 3-APs and 4-APs with bitmasks
    three_aps = []
    three_ap_masks = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            three_ap_masks.append(mask)

    four_aps = []
    four_ap_masks = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps.append((a, b, c, e))
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            four_ap_masks.append(mask)

    # Map elements to indices of APs containing them
    # element 1 is index 0 in the list, but we use 1-based indexing for elements
    # So elem_to_3aps[i] contains indices of 3-APs containing element i
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)

    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(mask):
        count = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                count += 1
        return count

    def is_valid_add(mask, x):
        """Check if adding x to set represented by mask creates a 4-AP."""
        bit = 1 << (x - 1)
        new_mask = mask | bit
        # Check all 4-APs containing x
        for ap_idx in elem_to_4aps[x]:
            if (new_mask & four_ap_masks[ap_idx]) == four_ap_masks[ap_idx]:
                return False
        return True

    def delta_add(mask, x):
        """Change in 3-AP count if we add x to set represented by mask."""
        delta = 0
        bit = 1 << (x - 1)
        for ap_idx in elem_to_3aps[x]:
            m = three_ap_masks[ap_idx]
            # The other two bits must be set in mask
            if (mask & (m ^ bit)) == (m ^ bit):
                delta += 1
        return delta

    def delta_remove(mask, x):
        """Change in 3-AP count if we remove x from set represented by mask (returns negative)."""
        lost = 0
        bit = 1 << (x - 1)
        for ap_idx in elem_to_3aps[x]:
            m = three_ap_masks[ap_idx]
            if (mask & (m ^ bit)) == (m ^ bit):
                lost += 1
        return -lost

    def is_valid_swap(mask, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        bit_out = 1 << (x_out - 1)
        bit_in = 1 << (x_in - 1)
        # New mask: remove x_out, add x_in
        new_mask = (mask & ~bit_out) | bit_in
        
        # Check all 4-APs containing x_in
        for ap_idx in elem_to_4aps[x_in]:
            if (new_mask & four_ap_masks[ap_idx]) == four_ap_masks[ap_idx]:
                return False
        return True

    def greedy_construct(seed_val):
        random.seed(seed_val)
        mask = 0
        available = list(range(1, N + 1))
        
        while available:
            candidates = []
            for x in available:
                if is_valid_add(mask, x):
                    d = delta_add(mask, x)
                    candidates.append((d, x))
            
            if not candidates:
                break
            
            # Sort by delta descending
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Pick best, with some randomness to diversify
            if random.random() < 0.8:
                best_x = candidates[0][1]
            else:
                top_k = min(10, len(candidates))
                best_x = random.choice([c[1] for c in candidates[:top_k]])
                
            mask |= (1 << (best_x - 1))
            available.remove(best_x)
            
        return mask

    def local_search(mask):
        """Hill climbing with 1-move adds and 1-1 swaps (first improvement)."""
        current_score = count_3aps(mask)
        improved = True
        
        while improved:
            improved = False
            
            # 1. Try adding elements
            for x in range(1, N + 1):
                bit = 1 << (x - 1)
                if mask & bit:
                    continue
                if not is_valid_add(mask, x):
                    continue
                gain = delta_add(mask, x)
                if gain > 0:
                    mask |= bit
                    current_score += gain
                    improved = True
                    # Break to restart loop as set changed
                    break
            
            if improved:
                continue

            # 2. Try 1-1 swaps (First Improvement)
            improved_swap = False
            # Iterate over elements in mask
            m = mask
            while m and not improved_swap:
                # Get lowest set bit
                lsb = m & -m
                x_out = lsb.bit_length()
                m ^= lsb
                
                loss = -delta_remove(mask, x_out)
                
                for x_in in range(1, N + 1):
                    if x_in == x_out:
                        continue
                    bit_in = 1 << (x_in - 1)
                    if mask & bit_in:
                        continue
                    
                    if not is_valid_swap(mask, x_out, x_in):
                        continue
                    
                    temp_mask = mask & ~(1 << (x_out - 1))
                    gain = delta_add(temp_mask, x_in)
                    
                    net = gain - loss
                    if net > 0:
                        mask = (mask & ~(1 << (x_out - 1))) | (1 << (x_in - 1))
                        current_score += net
                        improved_swap = True
                        improved = True
                        break
                        
        return mask, current_score

    def simulated_annealing(mask, t0=1.0, t1=0.01, steps=500):
        current_score = count_3aps(mask)
        best_mask = mask
        best_score = current_score

        for step in range(steps):
            if time.time() > deadline:
                break
            T = t0 * (t1 / t0) ** (step / steps)

            move_type = random.random()
            new_mask = mask
            new_score = current_score
            valid_move = False

            if move_type < 0.3:
                # Add
                candidates = [x for x in range(1, N + 1) if not (mask & (1 << (x - 1))) and is_valid_add(mask, x)]
                if candidates:
                    x = random.choice(candidates)
                    new_mask = mask | (1 << (x - 1))
                    new_score = current_score + delta_add(mask, x)
                    valid_move = True
            elif move_type < 0.5:
                # Remove
                # Pick a random element from mask
                m = mask
                if m:
                    # Pick random bit
                    # Simple way: convert to list? No, pick random index and check?
                    # Better: pick random number of bits to skip
                    # Or just pick random x in 1..N and check if in mask
                    # Since mask is sparse-ish, this is okay, but let's be smarter
                    # We can precompute a list of elements in mask for SA? 
                    # For now, let's just pick random x and hope it's in mask, or iterate
                    # Actually, for SA, we can just pick random x in 1..N
                    x = random.randint(1, N)
                    bit = 1 << (x - 1)
                    if mask & bit:
                        new_mask = mask & ~bit
                        new_score = current_score + delta_remove(mask, x)
                        valid_move = True
            else:
                # Swap
                # Pick random x_out in mask
                # Pick random x_in not in mask
                # To pick random x_out in mask efficiently without list conversion every time:
                # We can maintain a list of elements in mask, but that's expensive to update.
                # Let's just pick random x in 1..N for both and filter.
                x_out = random.randint(1, N)
                x_in = random.randint(1, N)
                bit_out = 1 << (x_out - 1)
                bit_in = 1 << (x_in - 1)
                
                if (mask & bit_out) and not (mask & bit_in):
                    if is_valid_swap(mask, x_out, x_in):
                        # Calculate delta
                        temp_mask = mask & ~bit_out
                        gain = delta_add(temp_mask, x_in)
                        loss = -delta_remove(mask, x_out)
                        delta = gain - loss
                        new_mask = (mask & ~bit_out) | bit_in
                        new_score = current_score + delta
                        valid_move = True

            if valid_move:
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask

        return best_mask

    best_mask = 0
    best_score = -1

    # Phase 1: Multiple greedy constructions + local search
    # Time-driven restarts
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        if time.time() > deadline - 10:
            break

        mask = greedy_construct(restart_count)
        score = count_3aps(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Local search is the main improver
        mask, score = local_search(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Short SA to escape local optima
        if time.time() < deadline - 5:
            # Slightly longer SA since local search is faster
            mask = simulated_annealing(mask, t0=0.5, t1=0.01, steps=800)
            score = count_3aps(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                
            # Another local search pass after SA
            mask, score = local_search(mask)
            if score > best_score:
                best_score = score
                best_mask = mask

    # Phase 2: Final polish on the best solution found
    if best_mask and time.time() < deadline - 2:
        mask, score = local_search(best_mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Low temp SA
        if time.time() < deadline - 2:
            mask = simulated_annealing(best_mask, t0=0.1, t1=0.001, steps=1500)
            score = count_3aps(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
            
        # Final local search
        mask, score = local_search(best_mask)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Convert mask back to sorted list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length()
        result.append(x)
        m ^= lsb
        
    return sorted(result)
