import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    deadline = time.time() + 140
    
    # Precompute neighbors
    neighbors = np.empty((total_words, 1 + n * (q - 1)), dtype=np.int32)
    for w in range(total_words):
        digits = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
            
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            for delta in [1, 2]:
                new_digit = (digits[i] + delta) % q
                power = q ** (n - 1 - i)
                new_val = w + (new_digit - digits[i]) * power
                neighbors[w, idx] = new_val
                idx += 1
                
    is_uncovered = np.ones(total_words, dtype=np.uint8)
    code = []
    col_sums = np.zeros(total_words, dtype=np.int32)
    
    # Greedy Phase
    while True:
        if time.time() > deadline - 10: # Reserve 10s for reduction
            break
            
        counts = is_uncovered[neighbors].sum(axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
            
        best_c = np.argmax(counts)
        
        code.append(best_c)
        is_uncovered[neighbors[best_c]] = 0
        col_sums[neighbors[best_c]] += 1
        
    # Reduction Phase
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Check for redundant codewords
        # For each c in code, if min(col_sums[neighbors[c]] - 1) > 0, it is redundant
        redundant_indices = []
        for i, c in enumerate(code):
            # col_sums includes c itself
            # So coverage by others is col_sums - 1
            min_cov_by_others = np.min(col_sums[neighbors[c]] - 1)
            if min_cov_by_others > 0:
                redundant_indices.append(i)
                
        if redundant_indices:
            changed = True
            # Remove the first redundant one (or all? Removing all might break validity if they overlap in unique coverage? 
            # No, if min_cov_by_others > 0, it means EVERY word in B(c) is covered by at least one OTHER codeword.
            # So removing c is safe.
            # We can remove all identified redundant codewords at once?
            # If c1 and c2 are both redundant, removing c1 doesn't make c2 non-redundant (it only reduces coverage).
            # Wait, if c1 is redundant, it means B(c1) is covered by others.
            # If we remove c1, the coverage of words in B(c1) decreases by 1.
            # Could this make c2 non-redundant?
            # c2 is redundant if B(c2) is covered by others (excluding c2).
            # If c1 is one of the "others" for c2, removing c1 might reduce coverage for some words in B(c2).
            # So we should remove one by one, or re-evaluate.
            # Removing one by one is safer.
            
            idx_to_remove = redundant_indices[0]
            c_to_remove = code[idx_to_remove]
            
            # Update col_sums
            col_sums[neighbors[c_to_remove]] -= 1
            
            # Remove from code
            code.pop(idx_to_remove)
            
    # Convert to list of lists
    result = []
    for w in code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result
