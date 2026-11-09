import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Generate RM(1,4) code extended to length 18
    # RM(1,4) length 16, size 32
    rm_words = []
    for c0 in range(2):
        for c1 in range(2):
            for c2 in range(2):
                for c3 in range(2):
                    for c4 in range(2):
                        word = 0
                        for x1 in range(2):
                            for x2 in range(2):
                                for x3 in range(2):
                                    for x4 in range(2):
                                        val = c0 ^ (c1 & x1) ^ (c2 & x2) ^ (c3 & x3) ^ (c4 & x4)
                                        # Pack into 16-bit integer
                                        # Order of bits: x1x2x3x4 -> 0..15
                                        idx = (x1 << 3) | (x2 << 2) | (x3 << 1) | x4
                                        if val:
                                            word |= (1 << idx)
                        # Extend to 18 bits by appending 00
                        # The 16 bits are in positions 0..15.
                        # Append 00 to positions 16, 17.
                        # So word is just the 16-bit value.
                        rm_words.append(word)
    
    best_words = rm_words[:]
    if time.time() > deadline:
        return best_words

    # Pre-compute bad masks
    bad_masks = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            bad_masks.append(m)
    
    # Try to extend the code
    # We will run multiple rounds of greedy addition
    for attempt in range(50):
        if time.time() > deadline - 2.0:
            break
            
        words = best_words[:]
        blocked = bytearray(1 << N)
        
        # Mark blocked
        for w in words:
            for m in bad_masks:
                blocked[w ^ m] = 1
        
        # Generate candidates
        # We want to find words not blocked
        # We can iterate through all 2^18 words
        # But that's 262k iterations. Fast.
        # We can also try random order
        
        # Try to add as many as possible
        # Use a random order for candidates to break symmetry
        candidates = list(range(1 << N))
        random.shuffle(candidates)
        
        added = 0
        for c in candidates:
            if time.time() > deadline - 2.0:
                break
            if blocked[c]:
                continue
            # Check if c is already in words (it shouldn't be if blocked is correct)
            # But blocked might have false positives? No, blocked is exact.
            # However, if c is in words, it is blocked.
            # So if not blocked, it is not in words and is valid.
            
            words.append(c)
            for m in bad_masks:
                blocked[c ^ m] = 1
            added += 1
            
        if len(words) > len(best_words):
            best_words = words[:]
            
    return best_words
