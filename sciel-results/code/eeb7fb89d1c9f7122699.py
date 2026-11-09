import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    best_words = [0]
    max_restarts = 1000 # Can be higher since each run is faster
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        words = [0]
        
        # Generate candidate order
        if i == 0:
            # Deterministic order for first run? Or just random.
            # Random is better for finding diverse good sets.
            candidates = list(range(1, 1 << N))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, 1 << N))
            random.shuffle(candidates)
            
        # Greedy construction
        for c in candidates:
            # Early exit if time is running out
            if time.time() > deadline:
                break
                
            # Check compatibility
            compatible = True
            for w in words:
                if (c ^ w).bit_count() < MIN_DIST:
                    compatible = False
                    break
            
            if compatible:
                words.append(c)
                
        # Update best
        if len(words) > len(best_words):
            best_words = words[:]
            
    return best_words
