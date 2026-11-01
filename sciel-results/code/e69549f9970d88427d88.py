"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    """
    Constructs a superpermutation of length n.
    For n <= 4, returns the known optimal (or near-optimal) short strings.
    For n >= 5, uses the standard recursive construction which yields a valid
    superpermutation of length sum_{k=1}^n k! + n - 1.
    
    The recursive construction works as follows:
    Let S_m be a superpermutation for symbols 1..m.
    To construct S_{m+1}, we iterate through all permutations P of 1..m in the
    order they appear as windows of length m in S_m.
    For each P, we append the block P + str(m+1) + P to the new word.
    We overlap the new block with the existing word as much as possible.
    
    This is the standard "P m P" construction.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Hardcoded optimal/near-optimal for small n
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction length: 1+2+6+24 = 33.
        # 123412314231243121342132413214321
        # Let's verify this string is valid and length 33.
        # Actually, let's just run the algorithm for n=4 to be sure, or hardcode if we are sure.
        # The algorithm is fast enough for n=4.
        pass

    # General recursive construction
    word = "1"
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, we should return a valid word.
            # However, the loop for n=7 is very fast (< 1s).
            # If we hit the deadline, it means something is wrong or n is huge.
            # For n <= 7, we should never hit this.
            # But to be safe, if we are in the middle of constructing, 
            # the current 'word' is a superpermutation for m-1 symbols, 
            # but not for n. 
            # Since n is small in this task (likely 7), we assume we finish.
            # If n were large, we would need a fallback.
            # For now, return the current state? No, it's invalid for n.
            # We must ensure we finish.
            pass

        # Extract unique (m-1) permutations in order of appearance from the current word
        perms = []
        seen = set()
        # The current word contains all permutations of 1..m-1 as windows of length m-1.
        # We need to extract them in the order they first appear.
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Check if it's a valid permutation (all distinct)
            # Since it comes from a superpermutation, it should be, but let's be safe.
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = ""
        for p in perms:
            # Block is P + str(m) + P
            block = p + str(m) + p
            
            # Find max overlap between suffix of new_word and prefix of block
            max_overlap = 0
            # The maximum possible overlap is min(len(new_word), len(block))
            # But we can be smarter. The overlap must be a suffix of new_word
            # and a prefix of block.
            # Since block starts with p (length m-1) and ends with p,
            # and new_word ends with the previous block's suffix.
            
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    max_overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
        # Check time again
        if time.time() > deadline:
            # If we are still not done, and n is small, this is unexpected.
            # But if n is large, we might need to bail.
            # The task says n is passed by evaluator.
            # If n is 7, it's fast.
            pass

    return word
# EVOLVE-BLOCK-END
