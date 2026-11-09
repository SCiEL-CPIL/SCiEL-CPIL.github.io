import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TARGET_DIM = 9
    TARGET_SIZE = 2 ** TARGET_DIM
    
    # Function to check if a vector v can be added to the code C
    # C is a list of integers
    def can_add(v, C):
        # Check distance from v to all c in C
        # If min distance >= MIN_DIST, return True
        min_d = N
        for c in C:
            d = (v ^ c).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    return False
        return True

    # Function to generate the code from a basis
    def generate_code(basis):
        words = [0]
        for b in basis:
            # Add b + w for all w in words
            new_words = [w ^ b for w in words]
            words.extend(new_words)
        return words

    # Greedy search for a basis of size 9
    best_code = []
    
    # Try multiple random restarts
    for attempt in range(50):
        random.seed(attempt)
        basis = []
        current_code = [0]
        
        # Candidate pool: all 2^18 vectors? Too many.
        # We can pick random vectors.
        # Or iterate through a subset.
        # Let's try picking random vectors.
        
        success = False
        for _ in range(TARGET_DIM):
            found = False
            # Try many random candidates
            for _ in range(1000):
                v = random.getrandbits(N)
                # Optional: ensure v is not in span(basis)
                # Simple check: if v is in current_code, skip
                if v in current_code:
                    continue
                
                if can_add(v, current_code):
                    basis.append(v)
                    # Update current_code
                    new_words = [w ^ v for w in current_code]
                    current_code.extend(new_words)
                    found = True
                    break
            
            if found:
                continue
            else:
                break
        
        if len(basis) == TARGET_DIM:
            # Verify the full code
            final_code = generate_code(basis)
            # Check min distance of final_code
            # Since we checked at each step, it should be valid.
            # But just in case, verify a sample or full check if small.
            # 512 words, 512*511/2 ~ 130k checks. Fast.
            valid = True
            for i in range(len(final_code)):
                for j in range(i+1, len(final_code)):
                    if (final_code[i] ^ final_code[j]).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return final_code
        
    # Fallback: Return the best code found or the seed
    # If we didn't find a 9-dim code, return the largest valid code found during attempts
    # Or just return the initial 32 words from the seed logic if needed?
    # The prompt asks to improve. If we fail to find 512, we should return something better than 32.
    # Let's keep the best partial code found.
    
    # Re-run to find the best partial code if we didn't return 512
    best_partial = []
    for attempt in range(10):
        random.seed(attempt + 100)
        basis = []
        current_code = [0]
        for _ in range(TARGET_DIM):
            found = False
            for _ in range(500):
                v = random.getrandbits(N)
                if v in current_code:
                    continue
                if can_add(v, current_code):
                    basis.append(v)
                    new_words = [w ^ v for w in current_code]
                    current_code.extend(new_words)
                    found = True
                    break
            if not found:
                break
        
        if len(current_code) > len(best_partial):
            best_partial = current_code
            
    if best_partial:
        return best_partial
    
    # Ultimate fallback: The original 32 words
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words
