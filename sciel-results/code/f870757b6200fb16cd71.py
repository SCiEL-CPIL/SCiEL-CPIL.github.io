import time
import random
import numpy as np

def construct(q, n, radius):
    # This implementation is specific to q=3, n=8, radius=1
    # as per the task description.
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task implies specific case
        # For now, return a trivial valid solution if possible or seed
        # The seed uses tetracode for n>=4.
        # For general case, a greedy might be needed, but we focus on the specific case.
        pass

    # Precomputation for q=3, n=8
    # Total words: 3^8 = 6561
    total_words = 3 ** n
    # Map word tuple to index
    # We can just use base-3 representation as index.
    # Index = sum(d_i * 3^(n-1-i))
    
    # Precompute neighbors for all 6561 words
    # neighbor[word_idx] is a list of indices of words at dist <= 1
    neighbors = [None] * total_words
    
    # Helper to convert index to list of digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % 3
            idx //= 3
        return digits

    # Helper to convert list of digits to index
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * 3 + d
        return idx

    # Precompute neighbors
    for i in range(total_words):
        digits = idx_to_digits(i)
        neigh = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val != orig:
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    neigh.append(digits_to_idx(new_digits))
        neighbors[i] = neigh

    # Initial code: Tetracode based
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Expand to length 8
    # For each tetracode word, and each tail of length 4
    initial_codewords = []
    for h in tetracode:
        for tail in np.ndindex(3, 3, 3, 3):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
            
    # Remove duplicates just in case (though construction should be unique)
    initial_codewords = list(set(initial_codewords))
    
    # State
    current_set = set(initial_codewords)
    counts = np.zeros(total_words, dtype=np.int32)
    
    # Initialize counts
    for c in current_set:
        for w in neighbors[c]:
            counts[w] += 1
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Local Search
    # We try to reduce size by removing redundant codewords.
    # If stuck, we do swaps to enable removals.
    
    # To speed up redundancy check, we can maintain a list of "potentially redundant" codewords?
    # Or just check all. 486 * 17 is small.
    
    while time.time() < deadline:
        # 1. Try to remove a redundant codeword
        removed = False
        # Shuffle to avoid bias
        codewords_list = list(current_set)
        random.shuffle(codewords_list)
        
        for c in codewords_list:
            # Check if c is redundant
            # Redundant if all w in neighbors[c] have counts[w] >= 2
            redundant = True
            for w in neighbors[c]:
                if counts[w] < 2:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                for w in neighbors[c]:
                    counts[w] -= 1
                current_set.remove(c)
                removed = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                break # Restart loop to re-evaluate
        
        if removed:
            continue
            
        # 2. No redundant codeword found. Try a swap.
        # Pick a random codeword to remove
        if not current_set:
            break
            
        c_old = random.choice(list(current_set))
        
        # Find critical words U
        U = []
        for w in neighbors[c_old]:
            if counts[w] == 1:
                U.append(w)
                
        if not U:
            # Should have been caught as redundant
            # But if counts changed? No, we just checked.
            # Maybe floating point? No, ints.
            # If U is empty, it IS redundant.
            # Force remove to avoid infinite loop?
            for w in neighbors[c_old]:
                counts[w] -= 1
            current_set.remove(c_old)
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            continue

        # Find candidates for c_new
        # Intersection of neighbors of all w in U
        cand_set = set(neighbors[U[0]])
        for w in U[1:]:
            cand_set.intersection_update(neighbors[w])
            
        # Exclude current codewords
        cand_set.difference_update(current_set)
        
        if cand_set:
            # Pick a random candidate
            c_new = random.choice(list(cand_set))
            
            # Perform swap
            # Remove c_old
            for w in neighbors[c_old]:
                counts[w] -= 1
            current_set.remove(c_old)
            
            # Add c_new
            for w in neighbors[c_new]:
                counts[w] += 1
            current_set.add(c_new)
            
            # Size is unchanged, so best_size not updated here
            # But the structure changed, might allow removals later
            
        else:
            # No valid swap for this c_old
            # We could try another c_old, but to keep it simple and fast,
            # we just loop and pick a new random one next iteration.
            # If we get stuck for too long, we might want to restart or change strategy.
            # For now, just continue.
            pass

    # Convert best_set back to list of lists of ints
    result = []
    for idx in best_set:
        result.append(idx_to_digits(idx))
        
    return result
