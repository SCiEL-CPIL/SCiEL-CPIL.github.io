import itertools
import random

def construct(q, n, radius):
    # Fallback logic remains the same
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Map tuple to index for fast lookup
    tuple_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbor_indices = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = tuple_to_idx[tuple(new_w)]
                    neighbor_indices[i].append(idx)
                    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(total_words):
            gain = 0
            for nb in neighbor_indices[i]:
                if nb in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        for nb in neighbor_indices[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    # Fast Pruning Phase
    # Calculate coverage count for each word
    coverage_count = [0] * total_words
    for idx in code:
        for nb in neighbor_indices[idx]:
            coverage_count[nb] += 1
            
    # Try to remove codewords
    # A codeword c can be removed if all its neighbors have coverage_count > 1
    # Note: This is a necessary and sufficient condition for *removable* in the context of 
    # "does removal cause any uncovered word?".
    # If min(coverage_count[nb] for nb in neighbors(c)) > 1, then removing c leaves all nb covered by others.
    
    changed = True
    while changed:
        changed = False
        new_code = []
        for idx in code:
            # Check if idx can be removed
            can_remove = True
            for nb in neighbor_indices[idx]:
                if coverage_count[nb] == 1:
                    # This neighbor is only covered by idx (or some other codeword we haven't removed yet? 
                    # No, coverage_count reflects the CURRENT code).
                    # If count is 1, removing idx makes it 0. So cannot remove.
                    can_remove = False
                    break
            
            if can_remove:
                # Remove idx
                for nb in neighbor_indices[idx]:
                    coverage_count[nb] -= 1
                changed = True
            else:
                new_code.append(idx)
        code = new_code
        
    return [words[i] for i in code]
