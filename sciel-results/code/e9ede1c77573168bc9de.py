import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # For the specific case q=3, n=7, radius=1
    # Total space size = 3^7 = 2187
    # We represent the universe of words as a set of integers 0..2186
    
    total_space = q ** n
    
    # Precompute neighbor masks
    # mask[i] is a Python int (bitmask) where bit j is 1 if word j is within distance radius of word i
    # Since total_space is 2187, we can use Python integers as bitmasks efficiently.
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    masks = [0] * total_space
    for j in range(total_space):
        m = 0
        m |= (1 << j)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    i = j + (new_val - orig_val) * powers[pos]
                    m |= (1 << i)
        masks[j] = m

    full_mask = (1 << total_space) - 1
    
    def get_word(idx):
        word = [0] * n
        val = idx
        for pos in range(n):
            word[pos] = val % q
            val //= q
        return word

    def get_idx(word):
        idx = 0
        for pos in range(n):
            idx += word[pos] * powers[pos]
        return idx

    # Strategy:
    # 1. Initialize with a good starting point.
    #    A simple greedy set cover is a standard start, but we can do better.
    #    Let's use a randomized greedy to get a valid cover.
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        remaining = full_mask
        selected = set()
        
        # To speed up greedy, we maintain a list of candidates.
        # However, recalculating gain for all 2187 items is fast enough if done smartly.
        # Gain of adding w = (masks[w] & remaining).bit_count()
        
        while remaining and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1
            
            # Optimization: If remaining is small, we might just brute force or pick arbitrary.
            # But let's stick to max gain.
            # We can iterate through all words not in selected.
            # To avoid O(N^2) in worst case, we can just scan all N words. N=2187 is small.
            for w in range(total_space):
                if w in selected:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1:
                break
                
            selected.add(best_word)
            remaining &= ~masks[best_word]
            
        return selected, remaining

    # Run a few greedy starts to find a decent initial valid cover
    best_initial = None
    best_initial_size = float('inf')
    
    for seed in range(10):
        sel, rem = greedy_cover(seed)
        if rem == 0:
            if len(sel) < best_initial_size:
                best_initial_size = len(sel)
                best_initial = sel
        
        if time.time() > deadline - 20:
            break
            
    if best_initial is None:
        # Fallback: just take all words? No, that's too big.
        # Greedy should always find a cover.
        best_initial, _ = greedy_cover(0)

    current_cover = set(best_initial)
    best_cover = set(current_cover)
    best_size = len(current_cover)
    
    # Local Search Phase
    # Moves:
    # 1. Remove redundant codewords (if removing w doesn't uncover any word).
    # 2. Swap: Remove w_out, Add w_in.
    #    Condition: The set of words covered by the new set is a superset of the old set?
    #    Or simply: Validity is maintained (all words covered) and size is reduced or same?
    #    Actually, for covering codes, we want to minimize size.
    #    A swap is useful if it allows removing another codeword later, or if it reduces size directly?
    #    Usually, we try to swap to a configuration where a removal becomes possible.
    
    # Let's define a function to check if a word is redundant in a given set
    def is_redundant(w, cover_set):
        # w is redundant if every word in masks[w] is covered by some other word in cover_set
        # unique_to_w = masks[w]
        # for x in cover_set:
        #     if x != w:
        #         unique_to_w &= ~masks[x]
        # return unique_to_w == 0
        
        # Optimization: Compute union of others?
        # union_others = 0
        # for x in cover_set:
        #     if x != w:
        #         union_others |= masks[x]
        # return (masks[w] & ~union_others) == 0
        
        # The first method (intersection of complements) is often faster if cover is large?
        # No, union is OR, intersection of complements is AND of NOTs.
        # Let's use the union approach but careful with performance.
        
        # Actually, let's precompute nothing, just iterate.
        # S is around 150-200. N is 2187.
        # 200 * (2187/64) ops is very fast.
        
        uncovered_by_others = 0
        for x in cover_set:
            if x != w:
                uncovered_by_others |= masks[x]
        
        return (masks[w] & ~uncovered_by_others) == 0

    def try_remove_redundants(cover_set):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for w in list(cover_set):
                if is_redundant(w, cover_set):
                    cover_set.remove(w)
                    changed = True
                    # Restart loop to catch new redundancies
                    break
        return cover_set

    # Initial cleanup
    current_cover = try_remove_redundants(current_cover)
    best_cover = set(current_cover)
    best_size = len(current_cover)
    
    # Local Search Loop
    # We will perform random swaps and check if they lead to a smaller cover after cleanup.
    # To make it deterministic-ish but effective, we use a time budget.
    
    all_words = list(range(total_space))
    
    while time.time() < deadline - 10:
        # Pick a random word to remove
        if not current_cover:
            break
        w_out = random.choice(list(current_cover))
        
        # Pick a random word to add
        w_in = random.choice(all_words)
        if w_in in current_cover:
            continue
            
        # Perform swap
        new_cover = set(current_cover)
        new_cover.remove(w_out)
        new_cover.add(w_in)
        
        # Check if new_cover is a valid cover?
        # If it's not valid, we can still keep it if we plan to add more? 
        # No, let's restrict to valid covers for simplicity and robustness.
        # Check validity: union of masks == full_mask
        union_mask = 0
        for w in new_cover:
            union_mask |= masks[w]
            
        if union_mask != full_mask:
            # Not a valid cover. Skip.
            # Alternative: Allow invalid covers and penalize? 
            # For now, strict validity keeps the search space smaller and ensures we always have a valid solution to return.
            continue
        
        # If valid, try to reduce it
        reduced_cover = try_remove_redundants(new_cover)
        
        if len(reduced_cover) < best_size:
            best_size = len(reduced_cover)
            best_cover = set(reduced_cover)
            current_cover = set(reduced_cover)
            
        # If size is same, we might still accept it to escape local minima?
        # Let's accept if size is <= best_size with some probability to explore.
        elif len(reduced_cover) == best_size and random.random() < 0.1:
            current_cover = set(reduced_cover)
            
        # If size increased, revert
        else:
            current_cover = set(current_cover) # Restore original

    # Final polish
    best_cover = try_remove_redundants(best_cover)
    
    # Convert to output format
    best_words = []
    for idx in best_cover:
        best_words.append(get_word(idx))
        
    # Verify validity one last time
    covered = 0
    for w in best_words:
        idx = get_idx(w)
        covered |= masks[idx]
        
    if covered != full_mask:
        # Fallback to greedy
        sel, _ = greedy_cover(0)
        best_words = [get_word(idx) for idx in sel]

    return best_words
