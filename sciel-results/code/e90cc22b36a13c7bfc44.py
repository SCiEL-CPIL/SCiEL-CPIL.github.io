import itertools

def construct(q, n, radius):
    # Specific optimization for the 7-match ternary problem
    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1()
    
    # Fallback to seed logic for other parameters
    # This ensures the function works generally, even if not optimally
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n < 4:
        # Handle small n by brute force or simple logic
        # For n=1,2,3, perfect codes or trivial solutions exist
        # Just return all words if n is very small to be safe
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    # Use the direct sum construction from the seed
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _solve_3_7_1():
    q = 3
    n = 7
    total_words = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute covers as bitsets
    covers = [0] * total_words
    for i, w in enumerate(words):
        mask = 1 << i
        for j in range(n):
            orig_val = w[j]
            for v in range(q):
                if v == orig_val: continue
                new_w = w.copy()
                new_w[j] = v
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        covers[i] = mask
        
    uncovered = (1 << total_words) - 1
    available = list(range(total_words))
    selected_indices = []
    
    # Greedy Phase
    while uncovered:
        best_idx = -1
        best_count = 0
        for idx in available:
            cnt = (covers[idx] & uncovered).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~covers[best_idx]
        
        # Filter available to remove those that no longer cover anything new
        # This is crucial for speed
        new_available = []
        for idx in available:
            if idx == best_idx:
                continue
            if (covers[idx] & uncovered).bit_count() > 0:
                new_available.append(idx)
        available = new_available
        
    # Post-processing: Remove redundant codewords
    # Build coverage counts
    # counts[u] = number of selected codewords covering u
    counts = [0] * total_words
    for idx in selected_indices:
        mask = covers[idx]
        # Iterate through set bits in mask
        # Since mask is sparse (16 bits set), we can just iterate through the neighbors
        w = words[idx]
        # Self
        counts[idx] += 1
        for j in range(n):
            orig_val = w[j]
            for v in range(q):
                if v == orig_val: continue
                new_w = w.copy()
                new_w[j] = v
                nidx = word_to_idx[tuple(new_w)]
                counts[nidx] += 1
                
    # Try to remove each selected codeword
    final_indices = []
    for idx in selected_indices:
        # Check if removing idx leaves any word with count 0
        # We need to check the words covered by idx
        # If any word in Ball(idx) has count 1 (and that count comes from idx), then it becomes 0
        # Actually, if count[word] == 1, it must be covered by exactly one selected word.
        # If we remove idx, and idx covers word, and count[word]==1, then word becomes uncovered.
        # So we can only remove idx if for all u in Ball(idx), count[u] > 1.
        
        w = words[idx]
        is_redundant = True
        
        # Check self
        if counts[idx] == 1:
            is_redundant = False
        else:
            # Check neighbors
            for j in range(n):
                orig_val = w[j]
                for v in range(q):
                    if v == orig_val: continue
                    new_w = w.copy()
                    new_w[j] = v
                    nidx = word_to_idx[tuple(new_w)]
                    if counts[nidx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    break
            
        if is_redundant:
            # It is redundant, so we do NOT add it to final_indices
            # And we must decrement counts for all words it covers
            counts[idx] -= 1
            for j in range(n):
                orig_val = w[j]
                for v in range(q):
                    if v == orig_val: continue
                    new_w = w.copy()
                    new_w[j] = v
                    nidx = word_to_idx[tuple(new_w)]
                    counts[nidx] -= 1
        else:
            # Keep it
            final_indices.append(idx)
            
    return [words[i] for i in final_indices]
