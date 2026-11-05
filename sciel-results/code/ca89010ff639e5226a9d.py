import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Use the direct sum method for general case, though it might be suboptimal
        # For q=3, n=4, radius=1, tetracode is perfect.
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic fallback: just return all words (worst case) or a simple product
        # This task is specific to 3, 8, 1.
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    # Precompute the neighborhood (Hamming distance <= radius) for every possible word
    # Since q=3, n=8, radius=1, the neighborhood size is 1 + n*(q-1) = 1 + 8*2 = 17
    
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Generate all words and their neighborhoods
    # This might take a bit of memory but 6561 * 17 is tiny
    neighborhoods = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    
    for idx in range(total_words):
        # Convert idx to word
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        
        # Generate neighbors
        neigh = [idx]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word.copy()
                    new_word[pos] = val
                    neigh.append(word_to_idx(new_word))
        
        neighborhoods[idx] = neigh
        
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    # To allow for randomized restarts if time permits
    best_code = None
    best_len = float('inf')
    
    num_restarts = 10 # We can do more if fast
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        current_code = []
        
        # Optimization: Pre-compute the "mask" of points covered by each word
        # neighborhoods is already an array of indices
        
        while True:
            if time.time() > deadline:
                break
                
            # Calculate gain for each word: number of points it covers that are not yet covered
            # covered[neighborhoods] is a boolean array of shape (total_words, neigh_size)
            # We want to count False values (uncovered points)
            
            # This operation is vectorized
            # is_covered = covered[neighborhoods] # (6561, 17)
            # gains = np.sum(~is_covered, axis=1)
            
            # To save memory/time, we can do this in chunks or just rely on NumPy speed
            is_covered = covered[neighborhoods]
            gains = np.sum(~is_covered, axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Pick the word with max gain
            # If there are ties, pick randomly to allow different restarts to explore different paths
            candidates = np.where(gains == max_gain)[0]
            best_idx = np.random.choice(candidates)
            
            current_code.append(best_idx)
            
            # Update covered
            covered[neighborhoods[best_idx]] = True
            
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code.copy()
            
        # Check if we have time for local search or more restarts
        if time.time() > deadline - 5.0:
            break
            
    if best_code is None:
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to words
    result = []
    for idx in best_code:
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result
