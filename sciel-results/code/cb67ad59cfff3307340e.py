import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback algebraic construction (486 words)
    def get_fallback():
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n > 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        # For n < 4, simple brute force or small cases
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    if q != 3 or n != 8 or radius != 1:
        # This specific optimization is tuned for 3^8, radius 1
        # For other cases, fallback is safer or we can generalize
        # But the task is specifically K_3(8,1).
        # If parameters differ, use fallback.
        return get_fallback()

    total_words = q ** n
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices covered by word i
    
    # Helper to convert int to base-q digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors array
    # Shape: (total_words, 1 + n*(q-1))
    # For q=3, n=8, size is 1 + 8*2 = 17
    num_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new number
                # num = w + (new_val - orig_val) * (q ** pos)
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1
        
    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            # Return best effort so far if it's valid, otherwise fallback
            # Greedy set cover is always valid if it finishes.
            # If it times out mid-way, the set might not cover everything.
            # So we should probably return fallback if we don't finish.
            # However, the loop is very fast.
            # Let's check if we can finish.
            # If we are close to finishing, maybe continue?
            # Safer to return fallback if timeout.
            return get_fallback()

        # Calculate gain for each candidate
        # uncovered[neighbors] creates an array of shape (total_words, num_neighbors)
        # sum(axis=1) gives gain for each candidate
        gains = uncovered[neighbors].sum(axis=1)
        
        # Find candidate with max gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            # Should not happen if uncovered.sum() > 0
            break
            
        selected_indices.append(best_idx)
        
        # Update uncovered
        uncovered[neighbors[best_idx]] = 0

    # Convert selected indices back to codewords
    codewords = []
    for idx in selected_indices:
        codewords.append(int_to_digits(idx, n, q))
        
    return codewords
