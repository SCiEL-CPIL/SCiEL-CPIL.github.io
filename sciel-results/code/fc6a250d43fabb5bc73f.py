import time
import random

def construct():
    N = 17
    MIN_DISTANCE = 6
    ALL_WORDS = range(1 << N)
    
    # Generate the 32-word Reed-Muller code (RM(1,4) extended? No, RM(1,4) is length 16, distance 8.
    # The task says "First-order Reed-Muller code on 16 coordinates, extended by two zero bits".
    # Wait, the seed code in the prompt:
    # for slope in range(16):
    #   for intercept in range(2):
    #     word = 0
    #     for point in range(16):
    #       bit = ((slope & point).bit_count() + intercept) % 2
    #       word |= bit << point
    #     words.append(word)
    # This generates 32 words of length 16.
    # The task requires 17-bit words.
    # The seed program returns these 16-bit words?
    # "Each integer encodes one N-bit binary codeword".
    # If N=17, the seed words are 16-bit. They are valid 17-bit words with a leading 0.
    # Distance is preserved.
    
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
            
    # Ensure we are working with 17-bit space.
    # The words are < 2^16, so they are valid in 2^17 space.
    
    best_codes = list(rm_words)
    best_size = len(rm_words)
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    while time.time() < deadline:
        # Create a shuffled list of all possible words
        # To avoid generating the list every time (memory churn), 
        # we can shuffle a pre-made list? 
        # list(range(131072)) is ~1MB. 
        # Shuffling 131072 items takes ~10-20ms.
        order = list(ALL_WORDS)
        random.shuffle(order)
        
        current_codes = list(rm_words)
        
        # Greedy fill
        for w in order:
            # Check if w is valid against all current codes
            # Optimization: If w is in current_codes, skip? 
            # Distance to self is 0, so it will fail the check anyway.
            is_valid = True
            for c in current_codes:
                if (w ^ c).bit_count() < MIN_DISTANCE:
                    is_valid = False
                    break
            
            if is_valid:
                current_codes.append(w)
                
        if len(current_codes) > best_size:
            best_size = len(current_codes)
            best_codes = current_codes
            
    return best_codes
