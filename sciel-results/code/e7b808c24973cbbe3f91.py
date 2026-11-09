import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m
    
    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def score_mask(mask):
        valid_diffs = 0
        # Max d is such that a + 2d <= 512, so 2d <= 511 -> d <= 255
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def score_list(lst):
        if len(lst) != N:
            return -1
        mask = list_to_mask(lst)
        return score_mask(mask)

    # Generate algebraic candidates
    candidates = []
    
    # Family 1: Base-4 Cantor-like (digits 0,1), 5 digits
    # Elements are sums of distinct powers of 4: 4^0..4^4
    # Max sum = 1+4+16+64+256 = 341
    # Scale must be 1 to fit 32 elements in 512 with offset
    base4_powers = [1, 4, 16, 64, 256]
    base4_max = sum(base4_powers) # 341
    
    # Generate all 32 base combinations
    base4_combos = []
    for mask_bits in range(32):
        val = 0
        for k in range(5):
            if (mask_bits >> k) & 1:
                val += base4_powers[k]
        base4_combos.append(val)
        
    # Try all offsets for Base-4 family
    for offset in range(1, MAX_VAL - base4_max + 1):
        cand = [offset + v for v in base4_combos]
        if all(1 <= x <= MAX_VAL for x in cand):
            candidates.append(cand)
            
    # Family 2: Base-3 Cantor-like (digits 0,1), 5 digits
    # Elements are sums of distinct powers of 3: 3^0..3^4
    # Max sum = 1+3+9+27+81 = 121
    base3_powers = [1, 3, 9, 27, 81]
    base3_max = sum(base3_powers) # 121
    
    base3_combos = []
    for mask_bits in range(32):
        val = 0
        for k in range(5):
            if (mask_bits >> k) & 1:
                val += base3_powers[k]
        base3_combos.append(val)
        
    # Try scales and offsets for Base-3 family
    for scale in range(1, 5): # 1..4
        scaled_max = scale * base3_max
        if scaled_max > MAX_VAL:
            continue
        for offset in range(1, MAX_VAL - scaled_max + 1):
            cand = [offset + scale * v for v in base3_combos]
            if all(1 <= x <= MAX_VAL for x in cand):
                candidates.append(cand)
                
    # Family 3: Dense Interval
    candidates.append(list(range(1, N + 1)))
    candidates.append(list(range(MAX_VAL - N + 1, MAX_VAL + 1)))
    
    # Family 4: Two Blocks
    candidates.append(list(range(1, N // 2 + 1)) + list(range(MAX_VAL - N // 2 + 1, MAX_VAL + 1)))
    
    # Evaluate all candidates
    best_score = -1
    best_list = None
    best_mask = 0
    
    for cand in candidates:
        if len(cand) != N:
            continue
        # Ensure distinct
        if len(set(cand)) != N:
            continue
        mask = list_to_mask(cand)
        s = score_mask(mask)
        if s > best_score:
            best_score = s
            best_list = cand[:]
            best_mask = mask
            
    # If no valid candidate found (should not happen), fallback to random
    if best_list is None:
        best_list = random.sample(range(1, MAX_VAL + 1), N)
        best_mask = list_to_mask(best_list)
        best_score = score_mask(best_mask)
        
    # Simulated Annealing Refinement
    # Start from the best algebraic candidate
    current_list = best_list[:]
    current_mask = best_mask
    current_score = best_score
    
    T = 10.0
    cooling_rate = 0.999
    
    def get_random_not_in_set(s):
        while True:
            v = random.randint(1, MAX_VAL)
            if v not in s:
                return v

    # SA Loop
    while time.time() < deadline:
        # Move: Pick one element to remove, pick one random element to add
        # This is a "1-swap" move
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]
        new_val = get_random_not_in_set(set(current_list))
        
        if old_val == new_val:
            continue
            
        # Apply move
        current_list[idx] = new_val
        new_mask = list_to_mask(current_list)
        new_score = score_mask(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            current_mask = new_mask
            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                best_mask = current_mask
        else:
            # Revert
            current_list[idx] = old_val
            current_mask = best_mask # Wait, best_mask is not current_mask if we reverted?
            # Actually, current_mask should be reverted too.
            # But we don't have a "revert mask" easily without re-calculating or storing old mask.
            # Let's just re-calculate mask if we revert? No, that's slow.
            # Better: Store old_mask.
            # But I didn't store old_mask in the loop above.
            # Let's fix the logic:
            # We should calculate new_mask, if rejected, revert list.
            # We don't need to update current_mask if rejected, but we need to ensure it matches current_list.
            # Since we reverted current_list, current_mask (which was from the previous accepted state) is still valid for current_list.
            # So no action needed for mask on rejection.
            
        T *= cooling_rate
        
    return best_list
