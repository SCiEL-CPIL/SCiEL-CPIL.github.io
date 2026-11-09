import random

# Pre-compute bit counts for 17-bit integers
BIT_COUNTS = [0] * (1 << 17)
for i in range(1, 1 << 17):
    BIT_COUNTS[i] = BIT_COUNTS[i >> 1] + (i & 1)

def construct():
    N = 17
    MIN_DIST = 6
    ALL_WORDS = 1 << N
    
    # Generate the 32-word seed from the original logic, extended to N bits
    # The original logic used 16 bits. We will use the same words, which are < 2^16, so valid in 17 bits.
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
            
    # Ensure distinctness (they are distinct in the original logic)
    seed_set = set(seed_words)
    
    # Create a list of all other words
    # To save memory and time, we can generate candidates on the fly or use a list
    # List of all words except seed
    candidates = [w for w in range(ALL_WORDS) if w not in seed_set]
    
    best_code = seed_words[:]
    best_size = len(best_code)
    
    # Multiple restarts
    # We can try different sorting strategies
    strategies = [
        lambda w: (w.bit_count(), random.random()), # Weight then random
        lambda w: (abs(w.bit_count() - 8), random.random()), # Close to weight 8
        lambda w: random.random() # Pure random
    ]
    
    for restart in range(15): # 15 restarts
        # Choose a strategy
        if restart % 3 == 0:
            sort_key = strategies[0]
        elif restart % 3 == 1:
            sort_key = strategies[1]
        else:
            sort_key = strategies[2]
            
        # Sort candidates
        # Sorting 131k elements takes ~0.1s
        candidates.sort(key=sort_key)
        
        current_code = seed_words[:]
        current_set = seed_set.copy()
        
        # Greedy pass
        for w in candidates:
            # Check distance to all in current_code
            valid = True
            for c in current_code:
                if BIT_COUNTS[w ^ c] < MIN_DIST:
                    valid = False
                    break
            if valid:
                current_code.append(w)
                current_set.add(w)
                
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
    return best_code
