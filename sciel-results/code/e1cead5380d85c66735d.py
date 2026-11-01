import itertools

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute the spheres as bitmasks
    # sphere_masks[i] is an int where bit j is 1 if word j is within distance radius of word i
    sphere_masks = [0] * total_words
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all words' digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Generate spheres
    for i in range(total_words):
        mask = 0
        base_digits = all_digits[i]
        
        # We need to find all words within Hamming distance <= radius
        # This can be done by iterating over all subsets of positions to change
        # and all possible values for those positions
        
        # For small radius and n, this is efficient enough
        # Number of positions to change: 0 to radius
        for r in range(radius + 1):
            # Choose r positions to change
            for pos_subset in itertools.combinations(range(n), r):
                # For each position in the subset, try all q values
                # We need to generate all combinations of values for these positions
                # If r is small, this is fast.
                # If r is large, this explodes. But for covering codes, r is usually small (1 or 2).
                # For r=1, we just change one position.
                # For r=2, we change two positions.
                
                # To be safe and efficient, we can just iterate all words and check distance?
                # No, that's O(N^2). 
                # Generating neighbors is better if radius is small.
                
                # Let's use a recursive or product-based generation for the changed positions
                # If r=0: just the word itself
                # If r=1: for each pos, for each val != base[pos]
                
                # A simpler way for small N: just check distance for all j?
                # N=2187. N^2 = 4.7M. Distance check is O(N). Total 10^10? Too slow in Python.
                
                # So we MUST generate neighbors.
                # Let's implement neighbor generation carefully.
                
                if r == 0:
                    mask |= (1 << i)
                    continue
                
                # Generate all variations at positions pos_subset
                # The positions NOT in pos_subset keep their original value.
                # The positions IN pos_subset take any value in 0..q-1.
                
                # We can use itertools.product for the values at the changed positions
                # But we need to map them back to the full word.
                
                # Optimization: Precompute masks for each position?
                # Let's just do the product.
                values = range(q)
                for vals in itertools.product(values, repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    new_idx = digits_to_index(new_digits)
                    mask |= (1 << new_idx)
                    
        sphere_masks[i] = mask

    # Greedy Set Cover
    covered_mask = 0
    all_mask = (1 << total_words) - 1
    codewords = []
    
    # To speed up, we can keep a list of candidates that might still add coverage
    # But since N is small, we just scan all.
    
    while covered_mask != all_mask:
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            # How many new words does i cover?
            new_covered = sphere_masks[i] & (~covered_mask & all_mask)
            count = new_covered.bit_count() if hasattr(int, 'bit_count') else bin(new_covered).count('1')
            
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_count == 0:
            break # Should not happen if code is valid
            
        codewords.append(best_idx)
        covered_mask |= sphere_masks[best_idx]
        
    # Convert indices back to list of lists
    result = []
    for idx in codewords:
        result.append(index_to_digits(idx))
        
    return result
