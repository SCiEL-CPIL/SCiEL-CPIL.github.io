import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Leave margin for overhead
    
    # Precompute masks for all distances < MIN_DIST
    # masks_by_dist[d] contains all bitmasks with exactly d bits set
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    # Flatten masks for faster iteration
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
        
    # blocked[i] is True if word i is within distance < MIN_DIST of any codeword in the code
    # We use a bytearray for memory efficiency and speed
    blocked = bytearray(1 << N)
    
    code = []
    
    # Function to add a word to the code and block its neighbors
    def add_word(w):
        if blocked[w]:
            return False
        code.append(w)
        # Block all words within distance < MIN_DIST
        for mask in all_masks:
            blocked[w ^ mask] = 1
        return True

    best_code = []
    
    # Run multiple random trials
    # The number of trials is limited by time
    num_trials = 0
    while time.time() - start_time < time_limit:
        # Reset blocked
        blocked[:] = b'\x00' * (1 << N)
        code = []
        
        # Randomize the order of words
        # Using a simple random shuffle of indices
        # To save time, we can just pick random words instead of shuffling 131k items
        # But shuffling is O(N) and fast.
        words = list(range(1 << N))
        random.shuffle(words)
        
        # Start with 0? Or random?
        # Starting with 0 is a good heuristic for symmetry, but random might be better for diverse packing.
        # Let's try starting with 0 in some trials, random in others.
        if num_trials % 2 == 0:
            add_word(0)
        
        for w in words:
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
                
        if len(code) > len(best_code):
            best_code = code[:]
            
        num_trials += 1
        
    return best_code
