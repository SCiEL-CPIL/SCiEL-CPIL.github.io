import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Pre-compute all masks with Hamming weight <= MIN_DIST - 1
    # These are the "forbidden" offsets for any given codeword
    max_weight = MIN_DIST - 1
    neighbor_masks = []
    for i in range(2**N):
        if i.bit_count() <= max_weight:
            neighbor_masks.append(i)
            
    # Use a mark array to track blocked words without re-initializing
    # mark[v] = restart_id if v is blocked in restart_id, else 0 or old id
    mark = [0] * (2**N)
    current_id = 1
    
    best_words = []
    max_restarts = 1000 # Cap high, time is the limit
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        words = []
        
        # To ensure diversity, we can randomize the starting point or order.
        # However, scanning 0..2^N-1 is deterministic.
        # A simple greedy scan in order 0..2^N-1 is biased towards low weights.
        # To avoid bias, we can shuffle the candidate order.
        # But shuffling 262k items takes time.
        # Alternative: Randomly pick the first word, then scan?
        # Or just rely on many restarts.
        # Actually, a deterministic scan 0..2^N-1 will always produce the SAME code.
        # We MUST randomize the order to explore different solutions.
        
        # Efficient randomization:
        # Instead of shuffling the full list, we can use a random permutation generator
        # or just pick a random start and wrap around?
        # Wrapping around is not enough.
        
        # Let's use a random shuffle of the indices.
        # 262k shuffle is fast (~10-20ms).
        candidates = list(range(2**N))
        random.shuffle(candidates)
        
        # Greedy pass with blocked array
        for c in candidates:
            if mark[c] != current_id:
                # c is available
                words.append(c)
                # Block c and its neighbors
                for mask in neighbor_masks:
                    mark[c ^ mask] = current_id
                    
        # Update best
        if len(words) > len(best_words):
            best_words = words[:]
            
        # Increment ID for next restart
        current_id += 1
        
        # Early stop if we hit a theoretical bound (if known) or just let time run out
        # If we find 512, we might stop? No, try to beat it.
        
    return best_words
