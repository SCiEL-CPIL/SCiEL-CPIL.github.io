import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # For this specific task, q=3, n=6, radius=1
    # But we write it generally for the interface, though optimized for this case.
    
    total_words = q ** n
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks
    # Ball of radius 1: self + 1 change in 1 position
    ball_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = list(w)
                new_word[pos] = new_val
                j = word_to_idx[tuple(new_word)]
                mask |= (1 << j)
        ball_masks[i] = mask
        
    # Helper to check if a set of codewords (indices) covers all words
    # This is done via coverage_counts in the main loop, but for initial check:
    
    def get_initial_code(seed_val):
        random.seed(seed_val)
        uncovered = (1 << total_words) - 1
        code = []
        code_set = set()
        
        # Symmetry breaking: fix 0
        first_idx = 0
        code.append(first_idx)
        code_set.add(first_idx)
        uncovered &= ~ball_masks[first_idx]
        
        # Greedy
        while uncovered != 0:
            best_idx = -1
            best_count = -1
            # Optimization: We can pre-sort or just scan. Scanning 729 items is fast.
            for i in range(total_words):
                if i in code_set:
                    continue
                covered = uncovered & ball_masks[i]
                count = covered.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        return code, code_set

    def compute_coverage_counts(code):
        counts = [0] * total_words
        for c in code:
            mask = ball_masks[c]
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                counts[idx] += 1
                m ^= lb
        return counts

    def is_removable(c_idx, counts):
        mask = ball_masks[c_idx]
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            if counts[idx] <= 1:
                return False
            m ^= lb
        return True

    def try_remove(c_idx, code, code_set, counts):
        # Assumes is_removable is True
        code.remove(c_idx)
        code_set.remove(c_idx)
        mask = ball_masks[c_idx]
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            counts[idx] -= 1
            m ^= lb
        return True

    def try_swap(c_old, c_new, code, code_set, counts):
        # Check validity
        # Decrement old
        mask_old = ball_masks[c_old]
        m = mask_old
        temp_counts = counts[:] # Copy is expensive? 729 ints. 
        # Better: check on the fly
        
        # Check if removing c_old makes any count 0
        m = mask_old
        valid = True
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            if counts[idx] <= 1:
                valid = False
                break
            m ^= lb
            
        if not valid:
            return False
            
        # Check if adding c_new causes any issues? No, adding only increases counts.
        # But we must ensure c_new is not already in code.
        if c_new in code_set:
            return False
            
        # Perform swap
        code.remove(c_old)
        code_set.remove(c_old)
        code.append(c_new)
        code_set.add(c_new)
        
        # Update counts
        m = mask_old
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            counts[idx] -= 1
            m ^= lb
            
        mask_new = ball_masks[c_new]
        m = mask_new
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            counts[idx] += 1
            m ^= lb
            
        return True

    best_code = None
    best_size = float('inf')
    
    num_restarts = 50 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        code, code_set = get_initial_code(r)
        counts = compute_coverage_counts(code)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Deletions
            # Try to remove any codeword
            # Iterate in reverse to keep indices stable if we pop, but we use list.remove
            # We iterate over a copy of the list
            for c in reversed(code):
                if is_removable(c, counts):
                    try_remove(c, code, code_set, counts)
                    improved = True
                    break # Restart deletion phase
            
            if improved:
                continue
                
            # Phase 2: Swaps
            # Try random swaps
            # We want to escape local minima
            # Limit swaps per restart step to avoid infinite loops on bad swaps
            swap_success = False
            for _ in range(200): # Try 200 random swaps
                if time.time() > deadline - 5.0:
                    break
                    
                if len(code) == 0:
                    break
                    
                c_old = random.choice(code)
                c_new = random.randint(0, total_words - 1)
                
                if c_new in code_set:
                    continue
                    
                if try_swap(c_old, c_new, code, code_set, counts):
                    swap_success = True
                    break
            
            if swap_success:
                improved = True # Trigger deletion phase again
            else:
                # If no swap worked, we are stuck.
                # Break out of local search for this restart
                break
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Early termination if we hit the theoretical lower bound or known best
        if best_size <= 73:
             # We can stop if we hit 73, but maybe 72 is possible?
             # The prompt says best known is 73.
             # Let's continue a bit more to see if we can get 72.
             # But 73 is a very good score.
             pass

    if best_code is None:
        # Fallback: return the initial greedy code
        code, _ = get_initial_code(0)
        return [list(all_words[i]) for i in code]
        
    return [list(all_words[i]) for i in best_code]
