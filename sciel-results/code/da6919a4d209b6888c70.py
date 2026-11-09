import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Initial code: 32 words from RM(1,4) extended
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extend to 17 bits? The seed used 16 bits.
            # The task requires 17-bit words.
            # The seed program returned 16-bit words?
            # "17-bit words".
            # The seed code: `word |= bit << point` for `point in range(16)`.
            # So the seed words are 16-bit.
            # But the task says "17-bit words".
            # "Each integer encodes one N-bit binary codeword".
            # "N=17".
            # The seed program might be invalid or N is dynamic?
            # "N-bit binary codeword".
            # The task contract says "length 17".
            # The seed code uses 16 bits.
            # If I return 16-bit words, are they valid?
            # "integers in [0, 2**N)".
            # If N=17, 16-bit words are in [0, 2^16), which is a subset.
            # But Hamming distance is calculated on 17 bits?
            # "pairwise Hamming distance".
            # Usually, leading zeros are considered.
            # If I use 16-bit words, the 17th bit is 0 for all.
            # Distance is the same as 16-bit distance.
            # So the 32-word code is valid for N=17.
            
            words.append(word)
            
    # Now, greedy add
    word_set = set(words)
    
    # Try to add more
    while time.time() < deadline:
        # Generate random word
        w = random.getrandbits(17)
        if w in word_set:
            continue
        
        # Check distance
        ok = True
        for existing in words:
            if (w ^ existing).bit_count() < 6:
                ok = False
                break
        
        if ok:
            words.append(w)
            word_set.add(w)
            
    return words
