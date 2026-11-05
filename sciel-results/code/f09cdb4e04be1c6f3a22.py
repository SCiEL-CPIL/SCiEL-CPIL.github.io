"""
Binary error-correcting codes of length 17, distance >= 6.
Goal: maximize number of codewords (A_2(17,6)).

Strategy:
1. Start with a good seed code (the 256-word linear [17,8,6] code from shortening
   the extended quadratic-residue [18,9,6] code, or equivalently the first-order
   Reed-Muller code RM(1,4) extended appropriately).
2. Use a greedy set extension with precomputed popcount table for O(1) Hamming
   distance checks.
3. Attempt to add words that maintain minimum distance 6.
4. Use local search / swaps to try to improve the set size.

Key optimization: precompute popcount for all 2^17 integers to make distance
checks O(1). Use a bitset-based approach for fast distance checking.
"""

import time
import random

N = 17
MIN_DIST = 6
TIME_LIMIT = 85  # safety margin below 90s

def construct():
    start_time = time.time()
    
    # Precompute popcount table for all 2^17 values
    # This enables O(1) Hamming distance: dist(a,b) = popcount[a ^ b]
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def hamming(a, b):
        return popcount[a ^ b]
    
    def check_min_dist(word, code_words):
        """Check if word has distance >= MIN_DIST from all words in code_words."""
        for w in code_words:
            if popcount[word ^ w] < MIN_DIST:
                return False
        return True
    
    # Build seed code: the [17,8,6] code
    # This is the shortened extended quadratic-residue code.
    # Alternatively, use RM(1,4) = 32 words of length 16, then extend.
    # Better: use the known 256-word code structure.
    # The [17,8,6] code can be constructed as follows:
    # Take the [18,9,6] extended QR code and shorten by one coordinate.
    # 
    # A simpler approach: use the first-order Reed-Muller code RM(1,4) which has
    # 32 codewords of length 16 and minimum distance 8. We can extend to length 17
    # and then try to add more words.
    #
    # Actually, let's use a better seed. The task says the 256-word linear reference
    # is the [18,9,6] extended quadratic-residue code shortened by one coordinate.
    # Let me construct a good seed and then greedily extend.
    
    # Seed: use RM(1,4) extended to 17 bits with an extra parity-like bit
    # RM(1,4): all affine functions on F_2^4, 32 words, length 16, min dist 8
    # Extend by adding a 17th bit = 0 for all, giving length 17, min dist 8
    # Then we can try to add more words.
    
    # Better seed: use the actual [17,8,6] code structure
    # The [17,8,6] code: generator matrix can be derived from the QR code.
    # For now, let's use a simpler approach:
    # Start with RM(1,4) (32 words, dist 8 in 16 bits), pad to 17 bits,
    # then greedily add words.
    
    # Actually, let me use a more systematic approach.
    # The [17,8,6] code has 256 words. Let me try to construct it or use a
    # good subset and extend.
    
    # For now, start with the 32-word RM(1,4) code padded to 17 bits
    code_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit is 0
            code_words.append(word)
    
    # Now greedily try to add words
    # Use a priority-based approach: try words in order of how "far" they are
    # from existing code, or just iterate through all 2^17 words.
    
    # For efficiency, we'll iterate through all possible words and add those
    # that satisfy the minimum distance constraint.
    # To make this faster, we can use a sorted order or random order.
    
    # Let's try multiple random orders and keep the best result
    best_code = list(code_words)
    best_size = len(best_code)
    
    # Try greedy extension with different orderings
    all_words = list(range(1 << N))
    
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Shuffle the candidate words
        candidates = all_words[:]
        random.shuffle(candidates)
        
        current_code = list(code_words)
        
        for word in candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            if word in current_code:
                continue
            if check_min_dist(word, current_code):
                current_code.append(word)
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
    
    # Try local search: swap out one word for two new words
    # This can sometimes increase the set size
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT - 10:
        improved = False
        current_code = list(best_code)
        
        for i in range(len(current_code)):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try removing current_code[i] and see if we can add more words
            temp_code = current_code[:i] + current_code[i+1:]
            
            # Try to add back words that were previously rejected
            # or try new words
            added = 0
            for word in all_words:
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                if word == current_code[i]:
                    continue
                if word in temp_code:
                    continue
                if check_min_dist(word, temp_code):
                    temp_code.append(word)
                    added += 1
                    if added > 1:  # If we can add more than we removed
                        break
            
            if added > 1:
                best_code = temp_code
                best_size = len(best_code)
                improved = True
                break
    
    return best_code

# Pre-seed random for reproducibility
random.seed(42)
