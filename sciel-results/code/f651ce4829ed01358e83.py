import random
import time

# Pre-compute bit counts for 17-bit integers
BIT_COUNTS = [0] * (1 << 17)
for i in range(1, 1 << 17):
    BIT_COUNTS[i] = BIT_COUNTS[i >> 1] + (i & 1)

def construct():
    N = 17
    MIN_DIST = 6
    ALL_WORDS = 1 << N
    
    # Generate the 32-word seed
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
            
    seed_set = set(seed_words)
    
    # List of all other words
    # We will filter this pool
    all_candidates = [w for w in range(ALL_WORDS) if w not in seed_set]
    
    best_code = seed_words[:]
    best_size = len(best_code)
    
    # Time limit check
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Strategy: Random shuffle
        # We can also try sorting by weight, but random is often better for diversity
        current_pool = all_candidates[:]
        random.shuffle(current_pool)
        
        current_code = seed_words[:]
        
        # Initial filter: Remove candidates too close to any seed word
        # This is O(Pool * Seed)
        # Pool ~ 131k, Seed 32. 4M ops. Fast.
        filtered_pool = []
        for w in current_pool:
            valid = True
            for c in current_code:
                if BIT_COUNTS[w ^ c] < MIN_DIST:
                    valid = False
                    break
            if valid:
                filtered_pool.append(w)
                
        current_pool = filtered_pool
        
        # Greedy extension with filtering
        # Iterate until pool is empty or time runs out
        while current_pool and (time.time() - start_time < time_limit):
            # Take the first candidate
            w = current_pool[0]
            current_code.append(w)
            
            # Filter the rest of the pool
            # We can skip the first element as it's already processed
            # List comprehension is fast
            # dist(x, w) >= 6
            current_pool = [x for x in current_pool[1:] if BIT_COUNTS[x ^ w] >= MIN_DIST]
            
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
        # Optional: Local Search / Swap
        # Try to swap a codeword with a candidate that allows more others?
        # This is complex. Let's stick to restarts for now.
        
    return best_code
