"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Start with Reed-Muller code (32 words, distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Greedy extension: try to add new words
    # Use a set for fast lookup of existing words
    word_set = set(words)
    
    # Try all 2^18 possible words in a strategic order
    # First try words with low weight, then high weight, etc.
    # Actually, let's try a smarter approach: iterate through all 2^18 words
    # but that's 262144 words, and checking each against up to 512 existing
    # words is feasible within time budget.
    
    # For efficiency, we'll iterate through candidate words and check distance
    # to all existing words. We can optimize by early termination.
    
    # Try all possible 18-bit words
    for candidate in range(2**18):
        if time.time() > deadline:
            break
        
        if candidate in word_set:
            continue
        
        # Check if candidate has distance >= 6 from all existing words
        valid = True
        for w in words:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        
        if valid:
            words.append(candidate)
            word_set.add(candidate)
    
    return words
# EVOLVE-BLOCK-END
