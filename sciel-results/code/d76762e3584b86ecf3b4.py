import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills at 150s. We keep a strict margin to ensure return.
    # 10s safety margin is standard, but let's use 5s to be safe for Python overhead.
    deadline = start_time + 145.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    # For q=3, n=8, total_words=6561. This is fast.
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    # A ball of radius r around i contains all words within Hamming distance r.
    # For q=3, n=8, radius=1, ball size is 1 + 8*(3-1) = 17.
    # We represent the ball as a Python integer (bitmask) where bit j is 1 if word j is in the ball.
    
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    # Fast path for radius=1, which is the specific case for this task
    if radius == 1:
        for i in range(total_words):
            d = digits_of[i]
            mask = 1 << i  # self
            
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
            balls[i] = mask
    else:
        # General radius (slower, but n=8 is small)
        for i in range(total_words):
            d = digits_of[i]
            mask = 1 << i  # self
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
            balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to check if a set of indices covers the space
    def is_covered(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask == all_mask

    # Helper to get the cover mask
    def get_cover_mask(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask

    # Initial Construction: Greedy Set Cover
    # This gives a good starting point for Local Search.
    def greedy_construct():
        uncovered = all_mask
        code = []
        # To speed up, we can just iterate. 
        # 6561 items, bit_count is fast.
        while uncovered:
            if time.time() > deadline - 10.0:
                return code # Return partial if timeout, but greedy is fast
            
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: Only consider indices that add new coverage?
            # Actually, just scan all. 6561 is small.
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
            
            if max_cover == 0:
                break # Should not happen if balls cover self
                
            # Random tie-breaking to allow different starts if needed
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Local Search: Remove-If-Redundant and Swap-For-Gain
    # State: list of indices (valid cover)
    # Moves:
    # 1. Remove an element if the cover remains valid.
    # 2. Swap an element with another if it reduces the size? No, swap keeps size same.
    #    Actually, we want to minimize size. So "Swap" is only useful if it enables future removals.
    #    Better strategy: 
    #    - Try to remove random elements. If valid, keep it.
    #    - If stuck, try to swap a random element with a random other element 
    #      (even if it makes the cover "worse" in terms of redundancy, it might break symmetry).
    #    - Actually, a simpler and very effective heuristic for Set Cover is:
    #      1. Try to remove any element. If valid, remove it.
    #      2. If no element can be removed, we are in a local minimum.
    #      3. Perturbation: Remove K random elements (making the cover invalid), 
    #         then greedily add back elements to restore validity, preferring those that 
    #         cover the most uncovered spots. This "kick" helps escape local minima.
    
    def local_search(code, time_limit):
        code = code[:]
        cover = get_cover_mask(code)
        
        # 1. Aggressive Removal Phase
        # Try to remove elements in a random order until no more can be removed
        changed = True
        while changed:
            changed = False
            # Shuffle indices to avoid bias
            indices_to_check = list(range(len(code)))
            random.shuffle(indices_to_check)
            
            for idx in indices_to_check:
                if time.time() > time_limit:
                    return code
                c = code[idx]
                # Check if removing c leaves a valid cover
                # (cover & ~balls[c]) == all_mask
                if (cover & ~balls[c]) == all_mask:
                    # Remove it
                    cover &= ~balls[c]
                    code.pop(idx)
                    changed = True
                    # After removal, the "indices_to_check" list is stale, so break and restart shuffle
                    break
        
        return code

    def perturb_and_recover(code, k=5):
        # Remove k random elements
        code = code[:]
        cover = get_cover_mask(code)
        
        # Remove k random distinct elements
        indices_to_remove = random.sample(range(len(code)), min(k, len(code)))
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            c = code[idx]
            cover &= ~balls[c]
            code.pop(idx)
        
        # Now cover is invalid. Greedily add back to restore validity.
        # We want to add words that cover the most currently uncovered bits.
        # To avoid simply restoring the original state, we can bias the choice.
        # But standard greedy recovery is robust.
        
        while cover != all_mask:
            if time.time() > deadline - 10.0:
                break # Timeout during recovery, return best effort (might be invalid, caller must check)
            
            uncovered = all_mask ^ cover # Bits that are 0 in cover
            
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: Only consider words that cover at least one uncovered bit
            # Scanning all 6561 is fine.
            for i in range(total_words):
                # If i is already in code, adding it does nothing (set semantics)
                # But we are using a list. We should check if it's already there?
                # For efficiency, we can just calculate gain. If gain is 0, it's redundant.
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
            
            if max_cover == 0:
                break # Should not happen if balls cover self and we start from valid set minus k
                
            # Random choice among best to diversify
            best_idx = random.choice(candidates)
            
            # Add to code
            # Check if already in code? 
            # If we removed it, it's not in code. 
            # If we didn't remove it, gain would be 0 (since its bits are already in cover).
            # So it's safe to add.
            code.append(best_idx)
            cover |= balls[best_idx]
            
        return code

    # Main Loop: Iterated Local Search (ILS)
    
    # 1. Get initial solution
    initial_code = greedy_construct()
    best_code = local_search(initial_code, deadline - 20.0)
    best_size = len(best_code)
    
    # 2. ILS Loop
    current_code = best_code[:]
    
    # We will run multiple restarts/perturbations
    # Each iteration:
    #   a. Perturb current_code
    #   b. Local Search (Remove)
    #   c. If better, update best
    
    # To ensure we finish, we allocate time per iteration or just run until deadline.
    # Perturbation + Recovery + Local Search is fast (<< 1s usually for this size).
    
    while time.time() < deadline - 10.0:
        # Perturb
        # The amount of perturbation can be adaptive, but fixed small k is often good.
        # Let's try k=3 to 5.
        k = random.randint(3, 6)
        perturbed = perturb_and_recover(current_code, k)
        
        # Check if perturbed is valid (it should be if recovery finished)
        # If recovery didn't finish due to timeout, perturbed might be invalid.
        # We must ensure we only feed valid covers to local_search.
        # local_search assumes valid input? 
        # My local_search implementation checks `(cover & ~balls[c]) == all_mask`. 
        # If the input is invalid, this check might fail to remove things, 
        # but it won't make it *more* invalid. 
        # However, if we return an invalid cover, the final answer is wrong.
        # So we must verify validity before accepting a new best.
        
        # Local Search on perturbed
        ls_result = local_search(perturbed, deadline - 10.0)
        
        # Verify validity of ls_result
        # This is a full check, O(N^2/word_size) but N is small.
        # Actually, local_search maintains the cover mask, so we can trust it 
        # IF we pass the correct cover mask. 
        # But my local_search doesn't return the mask. 
        # Let's just re-calculate cover for the final candidate.
        
        if len(ls_result) < best_size:
            # Check validity
            mask = 0
            for i in ls_result:
                mask |= balls[i]
            if mask == all_mask:
                best_code = ls_result
                best_size = len(ls_result)
                current_code = ls_result[:]
            else:
                # Invalid, discard
                pass
        elif len(ls_result) == best_size:
            # Even if same size, we might want to update current_code to diversify?
            # Usually, keeping the best is fine. But if we are stuck, 
            # updating current_code to a different solution of same size might help.
            # Let's do it with some probability.
            if random.random() < 0.1:
                current_code = ls_result[:]
                
    # Final cleanup: Ensure best_code is minimal
    # Run local_search one last time on best_code to ensure no redundant words remain
    if best_code:
        final_code = local_search(best_code, deadline - 2.0)
        # Verify
        mask = 0
        for i in final_code:
            mask |= balls[i]
        if mask == all_mask and len(final_code) <= len(best_code):
            best_code = final_code
        else:
            # Fallback to previous best
            pass
            
    # Fallback if something went wrong
    if not best_code:
        best_code = initial_code
        # Clean up greedy result too
        changed = True
        while changed:
            changed = False
            cover = get_cover_mask(best_code)
            for i in range(len(best_code)):
                c = best_code[i]
                if (cover & ~balls[c]) == all_mask:
                    best_code.pop(i)
                    changed = True
                    break

    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]
