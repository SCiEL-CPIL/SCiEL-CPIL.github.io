import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0 # Safety margin
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    best_words = []
    
    # We will perform multiple greedy passes.
    # Pass 1: Deterministic order (0 to 2^N - 1)
    # Subsequent passes: Randomized order
    
    max_restarts = 10 # Cap on restarts, but time is the real limit
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        words = []
        
        # Generate candidate order
        if i == 0:
            candidates = range(2**N)
        else:
            # To avoid generating a huge list and shuffling it every time (memory/time),
            # we can use a random permutation or just random sampling.
            # A full shuffle of 2^18 is ~262k ints. 
            # random.shuffle is fast.
            candidates = list(range(2**N))
            random.shuffle(candidates)
            
        # Greedy construction
        for c in candidates:
            # Check if c can be added
            # Early exit if time is running out
            if time.time() > deadline:
                break
                
            valid = True
            # Check against all existing words
            for w in words:
                if (c ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                words.append(c)
        
        # Update best
        if len(words) > len(best_words):
            best_words = words[:]
            
        # Optional: If we found a very good code, we might stop early?
        # But more restarts might find better.
        # If we hit the theoretical upper bound, stop.
        # Upper bound approx 1500.
        if len(best_words) > 1400:
            break

    return best_words
