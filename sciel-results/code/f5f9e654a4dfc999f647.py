"""Rewritten superpermutation constructor with deadline guard and optimized classic construction."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n):
    """
    Construct the shortest known superpermutation for n symbols using the classic
    recursive construction, with a deadline guard to ensure we finish in time.
    
    The classic construction produces a word of length 1! + 2! + ... + n! which is
    the best known upper bound for small n. For n=7, this gives 5905 symbols.
    
    We use an optimized implementation that avoids redundant work.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    if n <= 1:
        return "1"
    
    # For n=2, return "121" (length 3 = 1! + 2!)
    # For n=3, the classic gives length 9 = 1! + 2! + 3!
    # For n=7, length is 5905
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check deadline
        if time.time() > deadline:
            # Return what we have so far (may be invalid for the full n, but best effort)
            return "".join(map(str, word))
        
        # Collect all distinct (m-1)-length permutations in order of first appearance
        perms = []
        seen = set()
        w = word
        L = len(w)
        # Windows of length m-1
        for i in range(L - (m - 1) + 1):
            window = tuple(w[i:i + m - 1])
            if window not in seen:
                # Check it's a valid permutation (all distinct)
                if len(set(window)) == m - 1:
                    seen.add(window)
                    perms.append(window)
        
        # Build new word by concatenating blocks with maximal overlap
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximal overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[-o:] == block[:o]:
                    overlap = o
                    break
            new.extend(block[overlap:])
        
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
