import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 18
    MIN_DIST = 6
    
    # 1. Initial Seed: Reed-Muller RM(1, 4) extended to 18 bits
    # RM(1, 4) has 32 words, length 16, distance 8.
    # We pad with 2 zeros.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Pad to 18 bits (bits 16 and 17 are 0)
            words.append(word)
            
    # Ensure uniqueness (RM construction is unique)
    words = list(set(words))
    
    # Helper to check if a candidate can be added
    def can_add(candidate):
        for w in words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    # Helper to check distance to all
    def min_dist(candidate):
        min_d = N
        for w in words:
            d = (candidate ^ w).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    break
        return min_d

    # 2. Greedy Random Search
    # We will try to add random words.
    # To make it more effective, we can bias towards words that are "far" from the existing set.
    # But simple random is a good start.
    
    # Optimization: Keep words in a list. As the list grows, checking takes longer.
    # We can periodically sort or use a spatial index, but for N=18, brute force is likely fine up to ~1000 words.
    
    random.seed(42)
    
    added_count = 0
    # We will try both pure random and mutation-based search
    
    # Phase 1: Pure Random
    while time.time() < deadline - 5.0:
        if len(words) > 500: # If we get very large, random becomes inefficient
             break
        candidate = random.getrandbits(N)
        if can_add(candidate):
            words.append(candidate)
            added_count += 1
            
    # Phase 2: Mutation-based Local Search
    # Try flipping 1, 2, or 3 bits of existing words
    # This is more likely to find neighbors in the Hamming space
    
    # To avoid checking the same candidates repeatedly, we can keep a set of tried candidates
    # But memory might be an issue. We can just rely on the `can_add` check which also ensures uniqueness 
    # if we check against the set of words. Wait, `can_add` checks distance. 
    # If distance is 0, it means it's already in the set (if we consider distance 0 < 6).
    # So `can_add` implicitly handles duplicates if we ensure we don't add distance 0.
    # Distance 0 < 6, so it returns False. Correct.
    
    while time.time() < deadline:
        # Pick a random existing word
        base_word = random.choice(words)
        
        # Try flipping 1, 2, or 3 bits
        # Number of possible flips: C(18,1) + C(18,2) + C(18,3) = 18 + 153 + 816 = 987
        # We will try a random subset of these
        
        # Generate random flips
        num_flips = random.choice([1, 2, 3])
        bits_to_flip = random.sample(range(N), num_flips)
        
        candidate = base_word
        for b in bits_to_flip:
            candidate ^= (1 << b)
            
        if can_add(candidate):
            words.append(candidate)
            
    return words
