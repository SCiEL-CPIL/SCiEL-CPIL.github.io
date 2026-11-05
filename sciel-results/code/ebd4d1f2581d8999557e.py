import numpy as np
import time

def _build_linear_code():
    # Construct [17,8,6] code
    # Using the [18,9,6] QR code shortening
    # ... implementation ...
    return list_of_256_ints

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 17
    MIN_DIST = 6
    
    # 1. Get base code
    base_words = _build_linear_code()
    current_words = set(base_words)
    
    # 2. Greedy Add
    # Generate all candidates
    all_candidates = list(range(2**N))
    np.random.shuffle(all_candidates)
    
    # Vectorized check for a batch
    # We will iterate and check one by one, but optimized
    # Or use a NumPy array for current words
    
    # Let's use a list for current words and convert to array for checks
    # Checking 1 candidate against K words:
    #   arr = np.array(current_words, dtype=np.uint32)
    #   xored = arr ^ candidate
    #   dists = xored.view(np.uint8).reshape(-1, 4).sum(axis=1) # No, bit count is better
    #   Actually, np.bit_count is available in newer NumPy?
    #   If not, use lookup table or bin(x).count('1')
    #   Python's int.bit_count() is very fast.
    
    # So, loop over candidates:
    for cand in all_candidates:
        if time.time() > deadline - 10:
            break
        if cand in current_words:
            continue
        
        # Check distance
        min_d = 99
        # Optimization: if we have many words, this loop is slow.
        # But 256-300 words is small.
        for w in current_words:
            d = (w ^ cand).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    break
        
        if min_d >= MIN_DIST:
            current_words.add(cand)
            
    # 3. Local Search / Swaps
    # Try to improve by swapping
    # We maintain a list of "near misses" or just try random swaps
    
    # Convert to list for indexing
    code_list = list(current_words)
    
    # Try random swaps: remove one, add two
    # This is O(N^2) per swap if not careful.
    # Better: Pick a random word to remove.
    # Then try to find 2 words that fit.
    
    # We can pre-compute a list of "almost valid" words?
    # Or just run a second greedy pass with a different order?
    
    # Let's try a simple hill climbing:
    # Randomly remove a word, then greedily add as many as possible.
    # If the final size is larger, keep.
    
    best_code = set(code_list)
    best_size = len(best_code)
    
    while time.time() < deadline - 5:
        # Create a perturbed version
        trial_code = set(best_code)
        
        # Remove 1-3 random words
        k_remove = np.random.randint(1, 4)
        to_remove = np.random.choice(list(trial_code), size=k_remove, replace=False)
        for w in to_remove:
            trial_code.remove(w)
            
        # Greedy add from random candidates
        # Use a random subset of candidates to speed up
        candidates = np.random.choice(2**N, size=1000, replace=False)
        for cand in candidates:
            if cand in trial_code:
                continue
            min_d = 99
            for w in trial_code:
                d = (w ^ cand).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d >= MIN_DIST:
                trial_code.add(cand)
                
        if len(trial_code) > best_size:
            best_code = trial_code
            best_size = len(best_code)
            print(f"Improved to {best_size}")
            
    return list(best_code)
