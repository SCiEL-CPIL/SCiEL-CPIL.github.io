import time
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115
    
    # Precompute powers of q
    pow_q = np.array([q**i for i in range(n)], dtype=np.int64)
    total = q**n
    
    # Generate all words as integers
    all_words = np.arange(total, dtype=np.int64)
    
    # Convert word integer to vector
    digits = np.zeros((total, n), dtype=np.int64)
    tmp = all_words.copy()
    for i in range(n):
        digits[:, i] = tmp % q
        tmp //= q
        
    # Precompute the list of words covered by each codeword
    # For radius 1, q=3, n=8: 1 + 8*2 = 17 words
    num_covered = 1 + n * (q - 1)
    cover_indices = np.zeros((total, num_covered), dtype=np.int64)
    
    for c in range(total):
        d = digits[c]
        idx = 0
        cover_indices[c, idx] = c
        idx += 1
        for i in range(n):
            for v in range(q):
                if v != d[i]:
                    cover_indices[c, idx] = c + (v - d[i]) * pow_q[i]
                    idx += 1
                    
    # Precompute bitmasks for each codeword
    # mask[c] is an integer where bit w is 1 if codeword c covers word w
    masks = [0] * total
    for c in range(total):
        m = 0
        for w in cover_indices[c]:
            m |= (1 << int(w))
        masks[c] = m
        
    # Greedy Set Cover using bitmasks
    uncovered_mask = (1 << total) - 1  # All bits set
    selected = []
    cover_count = np.zeros(total, dtype=np.int32) # Count of selected codewords covering each word
    
    # Initial best score
    best_score = -1
    
    for step in range(total):
        if time.time() > deadline - 30:
            break
            
        # Find the codeword that covers the most uncovered words
        best_c = -1
        max_count = 0
        
        # Iterate through all candidates
        # Optimization: We can skip candidates that are already selected? 
        # No, greedy usually picks distinct ones. But we should avoid re-picking.
        # Actually, if we pick c, we remove its coverage from uncovered_mask.
        # If we pick c again, count will be 0 (if it was fully covered) or low.
        # But we should just pick the max.
        
        for c in range(total):
            if c in selected: # This is O(N) check, slow. Use a boolean array.
                continue
                
        # Let's use a boolean array for selected to make check O(1)
        # But we need to re-initialize or manage it.
        
        # Revised Loop:
        # We will maintain `selected_set` for O(1) lookup.
        
        # Wait, the loop above is inefficient. Let's restructure.
        
        # Break out of the previous pseudo-code.
        
    # Restart Greedy Loop properly
    selected = []
    selected_bool = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = (1 << total) - 1
    
    for step in range(total):
        if time.time() > deadline - 30:
            break
            
        best_c = -1
        max_count = 0
        
        for c in range(total):
            if selected_bool[c]:
                continue
            # Count uncovered words covered by c
            count = (uncovered_mask & masks[c]).bit_count()
            if count > max_count:
                max_count = count
                best_c = c
                
        if max_count == 0:
            break # All covered
            
        # Select best_c
        selected_bool[best_c] = True
        selected.append(best_c)
        
        # Update cover_count and uncovered_mask
        m = masks[best_c]
        # Update cover_count for words in m
        # We need to iterate over the 17 words
        for w in cover_indices[best_c]:
            cover_count[int(w)] += 1
            
        # Update uncovered_mask
        uncovered_mask &= ~m
        
    # Local Search: Redundancy Removal
    # Check if any selected codeword is redundant
    # A codeword c is redundant if for all w in cover_indices[c], cover_count[w] > 1
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Iterate in reverse to handle indices safely
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            c = selected[i]
            is_redundant = True
            for w in cover_indices[c]:
                if cover_count[int(w)] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                for w in cover_indices[c]:
                    cover_count[int(w)] -= 1
                selected_bool[c] = False
                selected.pop(i)
                changed = True
                
    # Convert selected indices to codeword vectors
    if not selected:
        # Fallback: return all words (should not happen for valid covering)
        return [digits[i].tolist() for i in range(total)]
    
    selected_codes = [digits[c].tolist() for c in selected]
    
    return selected_codes
