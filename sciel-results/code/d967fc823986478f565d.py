import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    total_space = q ** n
    # Map integers to base-q tuples for final output
    # We will work with integers 0..total_space-1 representing the words
    
    # Precompute neighbors
    # neighbors[i] will be a bitmask of all words within Hamming distance 'radius' of i
    # Since q=3, n=8, radius=1, the neighborhood size is 1 + n*(q-1) = 1 + 8*2 = 17
    
    def int_to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits
    
    neighbor_masks = [0] * total_space
    
    # Precompute neighbor masks
    for i in range(total_space):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        
        mask = 1 << i
        # Add neighbors at distance 1
        for pos in range(n):
            orig_val = digits[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * (q ** pos)
                new_index = i + delta
                if 0 <= new_index < total_space:
                    mask |= (1 << new_index)
                    
        neighbor_masks[i] = mask

    uncovered = (1 << total_space) - 1
    code_indices = []
    
    # Greedy Set Cover with early termination check
    while uncovered != 0:
        best_word = -1
        max_cover = 0
        
        for w in range(total_space):
            cover_count = (uncovered & neighbor_masks[w]).bit_count()
            
            if cover_count > max_cover:
                max_cover = cover_count
                best_word = w
                if max_cover == 0:
                    break
                
        if best_word == -1:
            break
            
        code_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]
        
        if time.time() > deadline:
            break
    
    # Local search: remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords
    # We try to remove each codeword and check if coverage is still complete
    
    if time.time() < deadline - 5:  # Leave some time for safety
        # Build a coverage count for each word
        coverage_count = [0] * total_space
        for idx in code_indices:
            mask = neighbor_masks[idx]
            # Count bits in mask
            # We need to add 1 to each position set in mask
            # Efficient way: iterate through set bits
            m = mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                pos = lb.bit_length() - 1
                coverage_count[pos] += 1
                m ^= lb
        
        # Try to remove each codeword
        removed = []
        for i, idx in enumerate(code_indices):
            if time.time() > deadline:
                break
            mask = neighbor_masks[idx]
            # Check if all words in this mask have coverage_count > 1
            # If yes, removing this codeword still leaves them covered
            m = mask
            redundant = True
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                if coverage_count[pos] <= 1:
                    redundant = False
                    break
                m ^= lb
            
            if redundant:
                # Remove this codeword
                removed.append(i)
                # Decrement coverage counts
                m = mask
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    coverage_count[pos] -= 1
                    m ^= lb
        
        # Rebuild code_indices without removed indices
        if removed:
            remove_set = set(removed)
            code_indices = [idx for i, idx in enumerate(code_indices) if i not in remove_set]
    
    # Second pass: try to remove more after first removals
    if time.time() < deadline - 2:
        coverage_count = [0] * total_space
        for idx in code_indices:
            mask = neighbor_masks[idx]
            m = mask
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                coverage_count[pos] += 1
                m ^= lb
        
        removed = []
        for i, idx in enumerate(code_indices):
            if time.time() > deadline:
                break
            mask = neighbor_masks[idx]
            m = mask
            redundant = True
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                if coverage_count[pos] <= 1:
                    redundant = False
                    break
                m ^= lb
            
            if redundant:
                removed.append(i)
                m = mask
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    coverage_count[pos] -= 1
                    m ^= lb
        
        if removed:
            remove_set = set(removed)
            code_indices = [idx for i, idx in enumerate(code_indices) if i not in remove_set]
    
    # Convert indices to lists of integers
    result = []
    for idx in code_indices:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)
        
    return result
