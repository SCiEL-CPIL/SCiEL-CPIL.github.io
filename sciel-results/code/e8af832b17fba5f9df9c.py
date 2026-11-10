import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for safety
    
    # For this specific task, q=3, n=8, radius=1
    # General logic for small q^n
    if q ** n > 1000000:
        # Fallback to seed if space is too large for bitset approach
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    
    # Precompute neighbors for all words
    # neighbors[i] is a bitmask of words within 'radius' of word i
    # Word i is represented by its integer value in base q
    # We need to convert index to word and back, or just iterate
    
    # Helper to convert index to list of digits
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    # Helper to convert list of digits to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute the bitmask of covered words for each potential codeword
    # neighbors_mask[i] is an integer where bit j is 1 if word j is within radius of word i
    neighbors_mask = [0] * total_words
    
    # Generate all words
    all_words = []
    for i in range(total_words):
        all_words.append(idx_to_word(i))
        
    # For each word i, find all words within radius
    # This is O(total_words * q * n * radius) roughly, which is fine for 6561
    for i in range(total_words):
        w_i = all_words[i]
        mask = 0
        # Generate all words within Hamming distance <= radius
        # We can do this by iterating through all possible variations
        # But a more efficient way for small radius is to start with w_i and flip coordinates
        
        # Actually, since total_words is small (6561), we can just check distance for all j?
        # 6561 * 6561 = 43M operations. In Python, this might be slow (seconds).
        # Let's optimize neighbor generation.
        
        # Better: Generate neighbors explicitly.
        # For radius 1:
        # 1. The word itself
        # 2. Words with one coordinate changed
        
        # Let's just do the explicit generation for radius 1 as it's the standard case
        # If radius is general, we might need a more complex generator, but radius=1 is most common in these tasks.
        
        # For radius 1:
        # Start with w_i
        mask |= (1 << i)
        
        for pos in range(n):
            orig_val = w_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                w_new = w_i.copy()
                w_new[pos] = new_val
                j = word_to_idx(w_new)
                mask |= (1 << j)
        
        neighbors_mask[i] = mask

    best_code = None
    best_len = float('inf')
    
    # Greedy Set Cover
    def greedy_cover(init_codewords, time_limit):
        covered = 0
        code = []
        used = set()
        
        # Add initial codewords
        for cw in init_codewords:
            idx = word_to_idx(cw)
            if idx not in used:
                used.add(idx)
                code.append(cw)
                covered |= neighbors_mask[idx]
                
        while True:
            if time.time() > time_limit:
                break
                
            # Find best next codeword
            best_gain = -1
            best_idx = -1
            
            # Iterate through all possible codewords not yet used
            # Optimization: We can skip used ones
            for i in range(total_words):
                if i in used:
                    continue
                gain = (neighbors_mask[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if best_gain == 0:
                        break # No more gains possible? Or just no gain for this one, but maybe others have 0 too.
                        
            if best_gain <= 0:
                break
                
            # Add best_idx
            used.add(best_idx)
            code.append(all_words[best_idx])
            covered |= neighbors_mask[best_idx]
            
            # Early termination if all covered
            if covered == (1 << total_words) - 1:
                break
                
        return code

    # Try multiple random restarts
    # The greedy algorithm is deterministic given the starting set.
    # Starting with empty set leads to a specific local optimum.
    # Starting with random sets might find better ones.
    
    # First, try standard greedy (empty start)
    code = greedy_cover([], deadline - 5)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code
        
    # Try a few random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Pick a small random set of starting codewords
        init_size = random.randint(1, 5)
        init_idxs = random.sample(range(total_words), init_size)
        init_words = [all_words[i] for i in init_idxs]
        
        code = greedy_cover(init_words, deadline - 2)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
    # If we still have time, we could try a more sophisticated local search, 
    # but greedy is usually sufficient for a jump from 486 to ~400-450.
    
    if best_code is None:
        # Fallback to seed
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
