import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Generate the [17,8,6] shortened quadratic-residue code
    # The [18,9,6] extended QR code is cyclic with generator polynomial:
    # g(x) = x^9 + x^7 + x^6 + x^4 + x^3 + x + 1
    # Binary representation (bits 9,7,6,4,3,1,0): 0b1011011011
    g = 0b1011011011
    
    base_words = []
    # Generate all 512 codewords of the [18,9,6] code
    for msg in range(512):
        c = 0
        # Multiply message polynomial (deg < 9) by g(x) (deg 9)
        # Result has degree < 18
        for i in range(9):
            if msg & (1 << i):
                c ^= (g << i)
        
        # Shorten by fixing the 18th bit (index 17) to 0
        # If bit 17 is 0, we keep the lower 17 bits
        if not (c & (1 << 17)):
            word = c & ((1 << 17) - 1)
            base_words.append(word)
            
    # Ensure uniqueness (should be 256 distinct words)
    base_words = list(set(base_words))
    
    # Initialize the code list and bit matrix
    code_list = base_words[:]
    # Convert to numpy array for fast distance calculations
    # Shape: (num_words, 17)
    if code_list:
        code_bits = np.array([(w >> i) & 1 for w in code_list for i in range(17)], dtype=np.uint8).reshape(-1, 17)
    else:
        code_bits = np.empty((0, 17), dtype=np.uint8)

    # 2. Greedy Augmentation
    # Iterate through all 17-bit words
    # We check if the candidate is at Hamming distance >= 6 from all existing codewords
    
    # Precompute bit extraction for speed? 
    # Using list comprehension for w_bits is fast enough.
    
    for w in range(1 << 17):
        # Check time limit
        if time.time() > deadline:
            break
        
        # Convert candidate w to bit vector
        w_bits = np.array([(w >> i) & 1 for i in range(17)], dtype=np.uint8)
        
        # Calculate Hamming distances to all existing codewords
        # XOR and sum
        dists = np.sum(code_bits ^ w_bits, axis=1)
        
        # Check if minimum distance is at least 6
        if np.min(dists) >= 6:
            # Add the word
            code_list.append(w)
            # Update code_bits
            # We can just vstack
            code_bits = np.vstack((code_bits, w_bits))
            
            # Optional: Early termination if we reach a known high bound?
            # 258 is the known max for this construction. 
            # If we reach 258, we could stop, but maybe there are more?
            # The task says "largest set". 
            # If we find more, great. If not, we stop when no more can be added.
            # Since we iterate through ALL candidates, we will find all possible additions.
            # So we don't need to stop early unless time runs out.
            
    return code_list
