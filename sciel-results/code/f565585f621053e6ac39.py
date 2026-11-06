import numpy as np
import time
from itertools import combinations

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
    
    # blocked[i] is True if word i is within distance < MIN_DIST of any codeword in the code
    # We use a bytearray for memory efficiency and speed
    blocked = bytearray(1 << N)
    
    code = []
    
    # Function to add a word to the code and block its neighbors
    def add_word(w):
        if blocked[w]:
            return
        code.append(w)
        # Block all words within distance < MIN_DIST
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1
        
    # Start with 0
    add_word(0)
    
    # Iterate through all other words
    # Using increasing order
    for w in range(1, 1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            add_word(w)
            
    return code
