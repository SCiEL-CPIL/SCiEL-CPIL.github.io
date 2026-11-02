import time
import random

_DEADLINE = 140.0

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # For the specific task: q=3, n=7, radius=1
    total = q ** n
    all_mask = (1 << total) - 1
    
    # Precompute ball masks
    # Word index to digits
    def idx_to_digits(idx):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits[::-1]
        
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    ball_masks = [0] * total
    for idx in range(total):
        digits = idx_to_digits(idx)
        mask = 1 << idx
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for alt in range(q):
                    if alt == orig:
                        continue
                    new_idx = idx + (alt - orig) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[idx] = mask

    # Greedy Set Cover
    def greedy_cover():
        uncovered = all_mask
        selected = []
        remaining = list(range(total))
        
        while uncovered != 0:
            if time.time() > deadline - 10:
                break
            best_count = -1
            best_idx = -1
            # To add randomness, we can shuffle remaining occasionally?
            # Or just pick max.
            for c in remaining:
                covered = ball_masks[c] & uncovered
                cnt = covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = c
                    if cnt == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            remaining.remove(best_idx)
        return selected

    # Local Search
    def local_search(C):
        if not C:
            return C
            
        # counts[w] = number of codewords in C that cover w
        counts = [0] * total
        mask = 0
        for c in C:
            mask |= ball_masks[c]
            # Update counts
            # Iterating bits of ball_masks[c]
            b = ball_masks[c]
            while b:
                lsb = b & -b
                w = lsb.bit_length() - 1
                counts[w] += 1
                b ^= lsb
                
        # Verify mask is all_mask (should be if C is valid cover)
        if mask != all_mask:
            # If not, try to fix? Or just return.
            # Greedy should produce valid cover.
            pass
            
        best_C = list(C)
        best_size = len(C)
        
        while time.time() < deadline - 5:
            # Squeeze Phase
            removed_any = False
            # Try removing codewords. Order matters? Random order helps escape.
            indices_to_try = list(range(len(C)))
            random.shuffle(indices_to_try)
            
            for i in indices_to_try:
                c = C[i]
                # Check if c is essential
                # Essential if any w in ball(c) has counts[w] == 1
                # And specifically, if counts[w]==1, that w is ONLY covered by c.
                # So if there is any w in ball(c) with counts[w]==1, c cannot be removed.
                # Wait, if counts[w]==1, it means exactly 1 codeword covers w.
                # If that codeword is c, then removing c leaves w uncovered.
                # So c is removable iff NO w in ball(c) has counts[w] == 1.
                
                b = ball_masks[c]
                essential = False
                while b:
                    lsb = b & -b
                    w = lsb.bit_length() - 1
                    if counts[w] == 1:
                        essential = True
                        break
                    b ^= lsb
                    
                if not essential:
                    # Remove c
                    # Decrement counts
                    b = ball_masks[c]
                    while b:
                        lsb = b & -b
                        w = lsb.bit_length() - 1
                        counts[w] -= 1
                        b ^= lsb
                        
                    mask &= ~ball_masks[c]
                    C.pop(i)
                    removed_any = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = list(C)
                    # After removal, other codewords might become removable, 
                    # but we are iterating over a snapshot of indices.
                    # It's safe to continue, but the indices are now shifted.
                    # To be safe and simple, if we remove one, we can restart the squeeze loop.
                    # Or just break and restart outer loop.
                    break 
                    
            if removed_any:
                continue
                
            # Swap Phase
            # Try to swap a codeword c with a new word w_new
            # Such that the cover remains valid.
            # Condition: All words uniquely covered by c (counts==1) must be covered by w_new.
            
            swapped = False
            # Pick a random codeword to swap
            # Or iterate? Random is better for exploration.
            for _ in range(100): # Try 100 random swaps per squeeze failure
                if not C: break
                i = random.randint(0, len(C) - 1)
                c = C[i]
                
                # Find U_c: words in ball(c) with counts == 1
                mask_Uc = 0
                b = ball_masks[c]
                while b:
                    lsb = b & -b
                    w = lsb.bit_length() - 1
                    if counts[w] == 1:
                        mask_Uc |= lsb
                    b ^= lsb
                    
                if mask_Uc == 0:
                    # c is not essential, should have been removed in squeeze.
                    # Maybe squeeze logic missed it? Or counts changed?
                    # If mask_Uc is 0, c is redundant. Remove it.
                    # (This shouldn't happen if squeeze is perfect, but safe to handle)
                    # Just remove it.
                    b = ball_masks[c]
                    while b:
                        lsb = b & -b
                        w = lsb.bit_length() - 1
                        counts[w] -= 1
                        b ^= lsb
                    mask &= ~ball_masks[c]
                    C.pop(i)
                    swapped = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = list(C)
                    break
                
                # Try random w_new
                w_new = random.randint(0, total - 1)
                
                # Check if w_new covers mask_Uc
                if (mask_Uc & ~ball_masks[w_new]) == 0:
                    # Valid swap!
                    # Remove c
                    b = ball_masks[c]
                    while b:
                        lsb = b & -b
                        w = lsb.bit_length() - 1
                        counts[w] -= 1
                        b ^= lsb
                    mask &= ~ball_masks[c]
                    
                    # Add w_new
                    b = ball_masks[w_new]
                    while b:
                        lsb = b & -b
                        w = lsb.bit_length() - 1
                        counts[w] += 1
                        b ^= lsb
                    mask |= ball_masks[w_new]
                    
                    C[i] = w_new
                    swapped = True
                    break
                    
            if swapped:
                # After a swap, run squeeze again
                continue
            else:
                # Stuck. 
                # Option 1: Break and return best_C.
                # Option 2: Perturb. 
                # Let's try a few random restarts of the whole process if time allows.
                # For now, break local search for this instance.
                break
                
        return best_C

    # Main execution
    # 1. Greedy
    C = greedy_cover()
    
    # 2. Local Search
    C = local_search(C)
    
    # 3. If time remains, try another greedy with randomization?
    # Greedy is deterministic. To randomize, we can shuffle the candidate list 
    # or pick randomly from top-k.
    # Let's try 1 more run if time > 50s remaining.
    if time.time() < deadline - 50:
        # Randomized Greedy: 
        # Instead of strict max, pick randomly from candidates covering > 80% of max?
        # Or just shuffle the order of evaluation? No, greedy picks max.
        # Let's just run greedy again, it's deterministic so it will yield same result.
        # To get diversity, we need to change the greedy strategy.
        
        # Strategy: Randomized Greedy
        # At each step, pick a candidate from the top 10% of best covering candidates randomly.
        
        def randomized_greedy():
            uncovered = all_mask
            selected = []
            remaining = list(range(total))
            while uncovered != 0:
                if time.time() > deadline - 10:
                    break
                best_count = -1
                best_idx = -1
                # Find max count
                counts = []
                for c in remaining:
                    covered = ball_masks[c] & uncovered
                    cnt = covered.bit_count()
                    counts.append((cnt, c))
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = c
                if best_idx == -1:
                    break
                
                # Pick from top 5% or top 10, whichever is larger
                # Sort counts descending
                counts.sort(reverse=True)
                k = max(1, len(counts) // 20) # top 5%
                if best_count > 0:
                    # Filter candidates with count > 0.9 * best_count
                    threshold = best_count * 0.9
                    candidates = [c for cnt, c in counts if cnt >= threshold]
                    if candidates:
                        best_idx = random.choice(candidates)
                
                selected.append(best_idx)
                uncovered &= ~ball_masks[best_idx]
                remaining.remove(best_idx)
            return selected
        
        C2 = randomized_greedy()
        if C2:
            C2 = local_search(C2)
            if len(C2) < len(C):
                C = C2
                
    # Convert to lists of digits
    result = []
    for idx in C:
        digits = idx_to_digits(idx)
        result.append(digits)
        
    return result
